#!/usr/bin/env python3
"""mass_resolution.py - tri-jet mass resolution per jet-assignment method.

Every method here answers the same question - *which three jets go with which
other three* - and is scored on the same observable: the average of the two
tri-jet candidate masses, ``(m1 + m2) / 2``. What changes between methods is
only the assignment, so the spread of that average is a clean measure of how
much of the mass resolution each method costs.

Methods scored
--------------
* **Every model in the ``--config``**, run exactly as ``evaluate.py`` runs it -
  same ONNX sessions, same jet pool (comb_solver: leading 7; spanet: its
  ``num_jets``, normally 8), same index bookkeeping. They are imported from
  ``run3_mj_evaluator.evaluate`` rather than reimplemented, so this script
  cannot drift from production.
* **``Comb00`` - ``Comb09``**: all 10 unordered 3+3 partitions of the leading
  ``--pool-size`` (default 6) jets, enumerated canonically - jet 0 is always in
  the first triplet, and the partition index is the position of the other two
  jets joining it. These are the combinatorial floor: whatever the models buy,
  they buy relative to this spread.
* **``CombRandom``**: one of those 10 drawn at random per event (``--seed``).
  The honest "no information" baseline - a single Comb0N is not, because a
  fixed partition index correlates with pT ordering.
* **``TruthScoutingJet``**: the assignment the gen record says is right,
  applied to the reco jets, on the events where it is available. The
  combinatorics are gone; everything else - jet energy resolution, out-of-cone
  radiation, the underlying event the trijet swallows - is still there.
* **``TruthGenJet``**: the same six quarks matched to **GenJets** instead, and
  the tri-jets built from those. No detector and no HLT at all, so what remains
  is the parton-to-jet step itself. The gap between the two truth rows is the
  reconstruction's share of the width, and it is the smaller half: on the
  practice file 5.4% at gen level against 6.9% at reco, i.e. ~4% added in
  quadrature. Skip it with ``--no-gen-truth`` on an input with no GenJets.

Truth matching
--------------
The RPV signal decays ``go -> u d s`` through lambda''_112, so unlike the
ttbar matching in ``run3-mj-analyzer/src/run3_mj_analyzer/truth_matching.py``
there is no intermediate W and no charge handle: **both gluinos have pdgId
1000021**, because the gluino is a Majorana fermion. Splitting the six quarks
by the sign of the mother's pdgId - what the ttbar code does with top/antitop -
silently puts all six in one group here. They are split by **mother index**
instead: select ``status == 23`` quarks whose mother is a gluino, and group by
which gluino they point at. On the practice file that gives exactly 6 quarks
from exactly 2 distinct gluinos in 100.0% of events, so no event is lost at
this stage.

Each quark is then matched to its nearest reco jet, and the event counts as
truth-matched when all six land within ``--dr-max`` of six *distinct* jets.
That requirement, not the gen record, is what limits the matched sample
(measured on the practice file, 10,724 events):

    matching pool      all 6 within dR<0.4      ... and distinct jets
    leading 6 jets            35.4%                     23.2%
    leading 7 jets            49.8%                     36.4%
    leading 8 jets            56.3%                     42.5%
    all jets                  60.7%                     46.6%

Matching runs against **all** jets by default, since a model's answer is an
index into the full jet array and the truth has to live in the same space to be
comparable. The table above is also the reason the accuracy column is quoted
on the matched subset only: on the other half of the events there is no right
answer to be had, and averaging over them would just dilute every method
toward the same number.

Resolution is quoted against the gluino pole mass read out of the gen record
(``GenPart_mass`` of the 1000021 - 1000.0 GeV on the practice file), so no mass
point has to be passed in.

Two widths are reported, and they deliberately disagree:

* ``sigma_eff``, half the central 68% interval, **includes** the combinatorial
  tail. It is the honest answer to "how wide is this distribution".
* ``fit sig``, from a Gaussian fitted iteratively over +/-``--fit-nsigma``
  sigma of the peak, describes the **core** and ignores that tail. It is the
  number a mass-peak plot looks like it has.

The fit needs no minimizer: a Gaussian is a parabola in log space, so a
sqrt(N)-weighted quadratic least squares on ``ln(counts)`` gives the mean and
width in closed form. That keeps scipy out of this repo's dependencies. The
window is re-centred each iteration and the fit is marked ``*`` if it never
settles. A method whose ``fit sig`` is much below its ``sigma_eff`` has a
narrow core sitting on a wide shelf - which is exactly what a wrong-pairing
method looks like, so quote both or the comparison flatters the bad ones.

Usage
-----
    python scripts/mass_resolution.py \\
        ../example_root_files/slimmed_RPV_M1000_ScoutNano_RPV_M1000_1.root \\
        --config config/config.json -o mass_resolution_M1000.root

The input is one slimmed ROOT file, several, or a JSON listing them - either
the slimmer/evaluator fileset layout (``{dataset: {"files": {path: tree}}}``)
or the analyzer's dataset layout (``{"datasets": {name: [paths]}}``).

Output is a ROOT file of histograms plus a PDF beside it (``--plot`` to place
it elsewhere, ``--no-plot`` to skip): page 1 overlays every model against the
random baseline and the two truth floors, page 2 the ten fixed partitions, and
the rest are per-method panels with the fitted Gaussian drawn over the
histogram. Every curve is labelled with its fitted mass, width and accuracy.

matplotlib is imported lazily and is **not** a dependency of this repo - the
evaluator wheel is pip-installed on every condor worker and has no reason to
carry it. If it is missing the run still completes and writes the ROOT file;
make the PDF afterwards, from an environment that has matplotlib, with::

    python scripts/mass_resolution.py --plot-from mass_resolution_M1000.root

which re-reads the histograms and fit parameters out of the output file and
needs neither onnxruntime nor the input data.

``--exactly-six`` restricts to 6-jet events, where every method is choosing
from the same six jets and the comparison is strictly like-for-like. Without
it the models use their production pools (7 or 8 jets) while the Comb methods
use the leading 6, which is the deployed-vs-floor comparison rather than an
algorithm-vs-algorithm one. Only 15.7% of the practice file has exactly six
jets, so the two modes answer different questions - run both.
"""

import argparse
import json
import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import numpy as np
import uproot

# Make the package importable without `pip install -e .` (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# NOTE: run3_mj_evaluator.evaluate is imported inside main(), not here. It
# imports onnxruntime at module scope, and --plot-from has to run in an
# environment that has matplotlib but not necessarily onnxruntime - which is
# the normal situation, since the evaluator wheel deliberately ships neither
# matplotlib nor a plotting step to the condor workers.

#: Pythia status of the outgoing partons of the hard subprocess - the three
#: quarks the gluino decays to carry it. Same convention as the analyzer's
#: truth_matching.py, which documents why statusFlags is not available in
#: ScoutingNano.
HARD_STATUS = 23

#: The gluino. Majorana, so both of them share this pdgId and the two decay
#: groups can only be told apart by mother index.
GLUINO_PDGID = 1000021

JET_BRANCHES = ["ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
                "ScoutingPFJet_m"]
GENJET_BRANCHES = ["GenJet_pt", "GenJet_eta", "GenJet_phi", "GenJet_mass"]
GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
                "GenPart_pdgId", "GenPart_status", "GenPart_genPartIdxMother"]

#: Jets are padded to this many slots before matching. Above the observed
#: maximum multiplicity (18 on the practice file); jets beyond it would be
#: dropped from the truth match, so raise it rather than trim.
MAX_JETS = 32


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def resolve_inputs(inputs, tree_override=None):
    """Flatten ROOT paths / dataset JSONs into ``[(path, tree), ...]``.

    Accepts the slimmer and evaluator fileset layout
    ``{dataset: {"files": {path: tree}}}`` and the analyzer's
    ``{"metadata": {...}, "datasets": {name: [paths]}}``, so a list made for
    either half of the pipeline works here unchanged.
    """
    jobs = []
    for item in inputs:
        if not item.endswith(".json"):
            jobs.append((item, tree_override or "events"))
            continue
        with open(item) as f:
            blob = json.load(f)
        if "datasets" in blob:  # analyzer layout
            default = tree_override or blob.get("metadata", {}).get("tree", "events")
            for paths in blob["datasets"].values():
                jobs.extend((p, default) for p in paths)
            continue
        for entry in blob.values():  # slimmer/evaluator layout
            if not isinstance(entry, dict) or "files" not in entry:
                raise SystemExit(
                    f"{item}: expected a fileset JSON with 'files' or a dataset "
                    f"JSON with 'datasets'; got keys {sorted(blob)}"
                )
            for path, tree in entry["files"].items():
                jobs.append((path, tree_override or tree))
    if not jobs:
        raise SystemExit("No input files resolved.")
    return jobs


# ---------------------------------------------------------------------------
# Combinatorial baselines
# ---------------------------------------------------------------------------

def enumerate_partitions(n=6):
    """The unordered 3+3 partitions of ``n`` objects, canonically ordered.

    Object 0 is pinned to the first triplet, which both removes the
    group1/group2 double count and fixes a deterministic partition index: for
    n=6 the result is the 10 ways of choosing the two jets that join the
    leading jet. Returns ``(g1, g2)``, each ``(n_part, 3)`` int.
    """
    g1s, g2s = [], []
    for pair in combinations(range(1, n), 2):
        g1 = (0,) + pair
        g2 = tuple(j for j in range(n) if j not in g1)
        if len(g1) != 3 or len(g2) != 3:
            raise ValueError(f"{n} objects do not split into two triplets")
        g1s.append(g1)
        g2s.append(g2)
    return np.array(g1s, dtype=int), np.array(g2s, dtype=int)


# ---------------------------------------------------------------------------
# Truth matching
# ---------------------------------------------------------------------------

def collection_to_numpy(chunk, prefix, min_jets=0):
    """Padded ``(pt, eta, phi, e, mask)`` for a flat ``<prefix>_*`` collection.

    ``evaluate.chunk_to_numpy`` cannot be reused for GenJets: ``jet_subarrays``
    hardcodes the mass field as ``<prefix>_m``, but the slimmer writes
    ``GenJet_mass`` (``slim.py`` GEN_JET_BRANCHES), so it raises
    ``FieldNotFoundError`` on any slimmed file. This accepts either spelling.
    Reco jets still go through ``chunk_to_numpy`` so that path stays identical
    to production.
    """
    mass_field = f"{prefix}_mass" if f"{prefix}_mass" in chunk.fields else f"{prefix}_m"
    for field in (f"{prefix}_pt", f"{prefix}_eta", f"{prefix}_phi", mass_field):
        if field not in chunk.fields:
            raise SystemExit(
                f"Input has no '{field}'. The GenJet truth needs the slimmer's "
                f"{prefix}_* branches; run without --gen-truth on data."
            )
    n = ak.to_numpy(ak.num(chunk[f"{prefix}_pt"], axis=1))
    width = max(min_jets, int(n.max()) if len(n) else 0)

    def pad(name):
        return ak.to_numpy(
            ak.fill_none(ak.pad_none(chunk[name], width, axis=1, clip=True), 0.0)
        )

    pt, eta, phi, m = (pad(f) for f in
                       (f"{prefix}_pt", f"{prefix}_eta", f"{prefix}_phi", mass_field))
    mask = np.arange(width)[None, :] < n[:, None]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return pt, eta, phi, e, mask


def _delta_r(eta1, phi1, eta2, phi2):
    """ΔR with phi wrapped into (-pi, pi]. Broadcasting is the caller's job."""
    dphi = np.abs(phi1 - phi2) % (2.0 * np.pi)
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    return np.hypot(eta1 - eta2, dphi)


def truth_assignment(chunk, jet_eta, jet_phi, jet_valid, dr_max):
    """Gen-truth jet assignment for an RPV ``go -> qqq`` pair.

    Parameters
    ----------
    chunk : awkward.Array
        One uproot chunk carrying the ``GenPart_*`` branches.
    jet_eta, jet_phi : (N, J) float
        Jets in pT-descending order, padded.
    jet_valid : (N, J) bool
        True where the padded slot is a real jet.
    dr_max : float
        Match radius.

    Returns
    -------
    idx : (N, 2, 3) int
        Jet indices of the two truth tri-jets (garbage where ``ok`` is False).
    ok : (N,) bool
        All six quarks matched to six distinct jets.
    m_pole : (N,) float
        Gluino mass from the gen record; NaN if no gluino is present.
    """
    pdg = chunk["GenPart_pdgId"]
    status = chunk["GenPart_status"]
    mother = chunk["GenPart_genPartIdxMother"]

    # pdgId of each particle's mother; index -1 means "no mother".
    safe = ak.where(mother < 0, 0, mother)
    mother_pdg = ak.where(mother < 0, 0, pdg[safe])

    absid = abs(pdg)
    is_daughter = (
        (status == HARD_STATUS)
        & (absid >= 1) & (absid <= 5)
        & (abs(mother_pdg) == GLUINO_PDGID)
    )

    n_events = len(pdg)
    n_found = ak.to_numpy(ak.num(pdg[is_daughter], axis=1))

    def _pad6(arr):
        return ak.to_numpy(
            ak.fill_none(ak.pad_none(arr[is_daughter], 6, axis=1, clip=True), 0.0)
        )

    q_eta = _pad6(chunk["GenPart_eta"])
    q_phi = _pad6(chunk["GenPart_phi"])
    q_mother = _pad6(mother).astype(np.int64)

    # Majorana gluino: the two decay groups differ only by which gluino they
    # point at, so group by mother index. Group 0 is whichever gluino the first
    # quark in the record came from.
    group = (q_mother != q_mother[:, :1]).astype(int)
    balanced = (n_found == 6) & (group.sum(axis=1) == 3)

    # Nearest jet per quark: (N, 6, J) -> (N, 6).
    dr = _delta_r(q_eta[:, :, None], q_phi[:, :, None],
                  jet_eta[:, None, :], jet_phi[:, None, :])
    dr = np.where(jet_valid[:, None, :], dr, np.inf)
    nearest = dr.argmin(axis=2)
    nearest_dr = dr.min(axis=2)

    within = (nearest_dr < dr_max).all(axis=1)
    # Six distinct jets: sorting and differencing is the vectorized way to ask
    # whether any two quarks grabbed the same jet.
    ordered = np.sort(nearest, axis=1)
    distinct = (np.diff(ordered, axis=1) > 0).all(axis=1)
    ok = balanced & within & distinct

    # Pack into (N, 2, 3) by group. Only meaningful where ok.
    idx = np.zeros((n_events, 2, 3), dtype=np.int64)
    if ok.any():
        rows = np.nonzero(ok)[0]
        g = group[rows]
        n = nearest[rows]
        for slot in (0, 1):
            sel = g == slot
            # Exactly three per group where balanced, so the reshape is safe.
            idx[rows, slot] = n[sel].reshape(len(rows), 3)

    is_gluino = abs(pdg) == GLUINO_PDGID
    masses = ak.fill_none(
        ak.pad_none(chunk["GenPart_mass"][is_gluino], 1, axis=1, clip=True), np.nan
    )
    m_pole = ak.to_numpy(masses)[:, 0]

    return idx, ok, m_pole


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

class MethodResult:
    """Per-method accumulator: masses, responses, and assignment agreement.

    ``pool`` is how many leading jets this method chooses from (None = all).
    It sets the method's own accuracy ceiling: a model that only ever sees the
    leading 7 jets cannot find a truth partition that needs the 8th, so
    accuracy is only interpretable next to ``n_reachable``.
    """

    def __init__(self, label, pool=None):
        self.label = label
        self.pool = pool
        self.mass = []          # average candidate mass, per event
        self.response = []      # mass / gluino pole mass
        self.masym = []         # candidate mass asymmetry
        self.mass_matched = []  # same, on truth-matched events only
        self.n_correct = 0      # events where the assignment equals the truth
        self.n_matched = 0      # truth-matched events this method saw
        self.n_reachable = 0    # matched events whose truth is inside the pool

    def add(self, m_avg, m_pole, masym, truth_ok, correct, reachable=None):
        self.mass.append(m_avg)
        self.response.append(m_avg / np.where(m_pole > 0, m_pole, np.nan))
        self.masym.append(masym)
        self.mass_matched.append(m_avg[truth_ok])
        self.n_matched += int(truth_ok.sum())
        if correct is not None:
            self.n_correct += int(correct.sum())
        if reachable is not None:
            self.n_reachable += int(reachable.sum())

    def finalize(self):
        cat = lambda xs: (np.concatenate(xs) if xs else np.empty(0))
        self.mass = cat(self.mass)
        self.response = cat(self.response)
        self.masym = cat(self.masym)
        self.mass_matched = cat(self.mass_matched)


def same_partition(a1, a2, b1, b2):
    """Per event, do the two 3+3 assignments agree (ignoring triplet order)?"""
    a1s, a2s, b1s, b2s = (np.sort(x, axis=1) for x in (a1, a2, b1, b2))
    direct = (a1s == b1s).all(axis=1) & (a2s == b2s).all(axis=1)
    swapped = (a1s == b2s).all(axis=1) & (a2s == b1s).all(axis=1)
    return direct | swapped


def n_assignments(pool):
    """How many distinct 3+3 assignments a pool of ``pool`` jets admits.

    ``C(n,3) * C(n-3,3) / 2`` - choose the first triplet, then the second from
    what is left, halved because the two triplets are unordered. 10 for six
    jets, 70 for seven, 280 for eight. Its reciprocal is the accuracy a method
    would get by guessing, and since the pools differ between methods that
    baseline differs too: 10% for the Comb* methods but 1.4% for a comb_solver
    and 0.36% for an 8-jet SPANet. Comparing raw accuracies across pools
    without it reads a bigger pool as a worse algorithm.
    """
    if pool is None or pool < 6:
        return None
    return comb(pool, 3) * comb(pool - 3, 3) // 2


def truth_in_pool(truth_idx, order, k):
    """Per event, are all six truth jets inside the leading-``k`` jets?

    ``k`` of None means the method sees every jet, so the truth is always
    reachable.
    """
    if k is None:
        return np.ones(len(truth_idx), dtype=bool)
    pool = order[:, :k]
    flat = truth_idx.reshape(len(truth_idx), 6)
    return (flat[:, :, None] == pool[:, None, :]).any(axis=2).all(axis=1)


def sigma_eff(values):
    """Half the central 68% interval - a width that survives combinatorial
    tails, where a standard deviation mostly measures the wrong-pairing shelf."""
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return float("nan")
    lo, hi = np.quantile(clean, [0.16, 0.84])
    return 0.5 * (hi - lo)


def gaussian_fit(counts, edges, mu0, sigma0, nsigma=2.0, iters=8, min_bins=5):
    """Iterative Gaussian fit to a histogram's core. Returns a dict.

    A Gaussian is a parabola in log space - ``ln N(x) = ln A - (x-mu)^2 /
    (2 sigma^2)`` - so a quadratic least squares on ``ln(counts)`` recovers the
    parameters in closed form, with no minimizer and no scipy (which is not in
    this repo's environment). Bins are weighted by ``sqrt(N)``, since the
    Poisson error on ``ln N`` is ``1/sqrt(N)``; empty bins carry no information
    in log space and are dropped.

    The fit window is re-centred on the current ``(mu, sigma)`` each iteration
    and runs until it stops moving. That matters here because every
    distribution in this study sits on a broad combinatorial shelf: a fit over
    the full range would measure the shelf, not the peak. **These numbers are
    therefore a core width, deliberately blind to the wrong-pairing tail** -
    the opposite convention from ``sigma_eff``, which includes it. Quote both
    or the reader cannot tell which question was answered.

    Returns ``{"amp", "mu", "sigma", "nbins", "converged"}``, with NaN
    parameters if the core never resolved (too few filled bins, or a fitted
    parabola that curves the wrong way - i.e. no peak to speak of).
    """
    failed = {"amp": float("nan"), "mu": float("nan"), "sigma": float("nan"),
              "nbins": 0, "converged": False}
    centres = 0.5 * (edges[:-1] + edges[1:])
    mu, sigma, converged, nbins = float(mu0), float(sigma0), False, 0
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return failed

    for _ in range(iters):
        sel = (
            (centres > mu - nsigma * sigma)
            & (centres < mu + nsigma * sigma)
            & (counts > 0)
        )
        nbins = int(sel.sum())
        if nbins < min_bins:
            return failed
        x, y, w = centres[sel], np.log(counts[sel]), np.sqrt(counts[sel])
        try:
            a, b, c = np.polyfit(x, y, 2, w=w)
        except (np.linalg.LinAlgError, ValueError):
            return failed
        if a >= 0:            # opens upward: a dip, not a peak
            return failed
        new_mu = -b / (2.0 * a)
        new_sigma = np.sqrt(-1.0 / (2.0 * a))
        if not (np.isfinite(new_mu) and np.isfinite(new_sigma)) or new_sigma <= 0:
            return failed
        # Runaway guard: a window that walks off the histogram is a failed fit,
        # not a wide one.
        if new_mu < edges[0] or new_mu > edges[-1] or new_sigma > (edges[-1] - edges[0]):
            return failed
        converged = (abs(new_mu - mu) < 0.01 * sigma
                     and abs(new_sigma - sigma) < 0.01 * sigma)
        mu, sigma = new_mu, new_sigma
        if converged:
            break

    return {"amp": float(np.exp(a * mu * mu + b * mu + c)), "mu": float(mu),
            "sigma": float(sigma), "nbins": nbins, "converged": bool(converged)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _fit_label(label, fit, acc):
    """Legend text: the method, its fitted peak and width, and its accuracy."""
    if fit and np.isfinite(fit.get("mu", np.nan)):
        star = "" if fit.get("converged") else "*"
        core = f"m={fit['mu']:.0f}{star}, $\\sigma$={fit['sigma']:.0f}"
        if fit["mu"]:
            core += f" ({100 * fit['sigma'] / fit['mu']:.0f}%)"
    else:
        core = "no fit"
    return f"{label} — {core}" + (f", acc={acc}" if acc else "")


def make_plots(entries, pdf_path, mass_range, title=None):
    """Write the multi-page PDF. ``entries`` is a list of plot-ready dicts.

    matplotlib is imported here, not at module scope: the evaluator wheel is
    pip-installed on every condor worker, and plotting is an interactive step
    that no batch job needs, so it must not become a hard dependency. A missing
    matplotlib costs you the PDF and nothing else.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # write files; never needs a display
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("\n[plot] matplotlib is not installed in this environment, so no "
              "PDF was written.\n       Install it (pkg-env/bin/pip install "
              "matplotlib), or re-make the plot\n       from the ROOT file with "
              "an environment that has it:\n"
              "         python scripts/mass_resolution.py --plot-from "
              f"{pdf_path.with_suffix('.root')}")
        return False

    combs = [e for e in entries if e["key"].startswith("Comb")
             and e["key"] != "CombRandom"]
    headline = [e for e in entries if e not in combs]

    def draw(ax, group, density=True):
        for e in group:
            values, edges = e["values"], e["edges"]
            total = values.sum()
            if total <= 0:
                continue
            y = values / (total * np.diff(edges)) if density else values
            style = {}
            if e["key"] == "TruthScoutingJet":
                style = {"color": "k", "linewidth": 2.0}
            elif e["key"] == "TruthGenJet":
                style = {"color": "k", "linewidth": 2.0, "linestyle": "--"}
            ax.stairs(y, edges, label=_fit_label(e["label"], e["fit"], e["acc"]),
                      **style)
        ax.set_xlim(*mass_range)
        ax.set_xlabel("average tri-jet candidate mass [GeV]")
        ax.set_ylabel("a.u. (unit area)" if density else "events")
        ax.legend(fontsize=7.5, loc="upper right")

    with PdfPages(pdf_path) as pdf:
        # Page 1 - the comparison that matters: every model against the
        # random baseline and the two truth floors.
        fig, ax = plt.subplots(figsize=(9, 5.5))
        draw(ax, headline)
        ax.set_title(title or "Mass resolution by assignment method")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2 - the ten fixed partitions, i.e. how much of the spread is
        # pure combinatorics.
        if combs:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            draw(ax, combs)
            ax.set_title("The 10 six-jet partitions")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 3+ - one panel per method, raw counts with the fitted Gaussian
        # drawn on top. This is where a bad fit is visible rather than merely
        # flagged, so it is plotted in counts, the space the fit ran in.
        per_page = 9
        for start in range(0, len(entries), per_page):
            group = entries[start:start + per_page]
            fig, axes = plt.subplots(3, 3, figsize=(12, 9))
            for ax, e in zip(axes.flat, group):
                values, edges = e["values"], e["edges"]
                ax.stairs(values, edges, color="C0")
                fit = e["fit"]
                if fit and np.isfinite(fit.get("mu", np.nan)):
                    x = np.linspace(edges[0], edges[-1], 400)
                    ax.plot(x, fit["amp"] * np.exp(
                        -((x - fit["mu"]) ** 2) / (2 * fit["sigma"] ** 2)),
                        "r-", linewidth=1.2)
                    ax.axvline(fit["mu"], color="r", linestyle=":", linewidth=0.8)
                    note = f"m={fit['mu']:.0f}\n$\\sigma$={fit['sigma']:.0f}"
                    if not fit.get("converged"):
                        note += "\n(no conv.)"
                else:
                    note = "no fit"
                if e["acc"]:
                    note += f"\nacc={e['acc']}"
                ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top",
                        fontsize=7.5)
                ax.set_title(e["label"], fontsize=9)
                ax.set_xlim(*mass_range)
                ax.tick_params(labelsize=7)
            for ax in axes.flat[len(group):]:
                ax.axis("off")
            fig.supxlabel("average tri-jet candidate mass [GeV]", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"plots written to {pdf_path}")
    return True


def entries_from_results(results, n_truth):
    """Plot-ready dicts straight out of an in-memory run."""
    out = []
    for key, res in results.items():
        acc = ("" if key.startswith("Truth") or not n_truth
               else f"{100.0 * res.n_correct / n_truth:.1f}%")
        out.append({
            "key": key, "label": res.label, "fit": res.fit, "acc": acc,
            "values": res.hist.view()["value"],
            "edges": res.hist.axes[0].edges,
        })
    return out


def entries_from_file(root_path):
    """Plot-ready dicts read back from a previous run's output file.

    The point of this path is that the environment which can run the ONNX
    models is not necessarily the one with matplotlib: everything the plot
    needs - the histograms and the fit parameters - is already in the output
    file, so the PDF can be made later, elsewhere, without re-running anything.
    """
    with uproot.open(root_path) as f:
        summary = json.loads(str(f["summary"]))
        n_truth = summary.get("n_truth_matched", 0)
        out = []
        for key, meta in summary["methods"].items():
            h = f[f"h_mass_{key}"].to_boost()
            acc = ("" if key.startswith("Truth") or not n_truth
                   else f"{100.0 * meta.get('n_correct', 0) / n_truth:.1f}%")
            out.append({
                "key": key, "label": meta.get("label", key),
                "fit": meta.get("fit"), "acc": acc,
                "values": h.view()["value"], "edges": h.axes[0].edges,
            })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Tri-jet mass resolution per assignment method: the config's "
                    "models, all 10 six-jet partitions, a random one, and the "
                    "gen-truth assignment."
    )
    p.add_argument("inputs", nargs="*",
                   help="slimmed ROOT file(s), or a JSON listing them "
                   "(not needed with --plot-from)")
    p.add_argument("--config", default=None,
                   help="evaluator config JSON; every model in it is scored "
                   "(not needed with --plot-from)")
    p.add_argument("-o", "--output", default="mass_resolution.root",
                   help="output ROOT file of histograms (default: %(default)s)")
    p.add_argument("--tree", default=None,
                   help="input tree name (default: 'events', or the JSON's)")
    p.add_argument("--pool-size", type=int, default=6, metavar="N",
                   help="leading-N jets the Comb* baselines choose from; only "
                   "6 gives exactly 10 partitions (default: %(default)s)")
    p.add_argument("--exactly-six", action="store_true",
                   help="keep only events with exactly 6 jets, so the models "
                   "and the Comb* baselines see the same jets")
    p.add_argument("--dr-max", type=float, default=0.4, metavar="R",
                   help="quark-jet match radius (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for CombRandom (default: %(default)s)")
    p.add_argument("--mass-range", type=float, nargs=2, default=(0.0, 2000.0),
                   metavar=("LO", "HI"), help="mass histogram range in GeV")
    p.add_argument("--bins", type=int, default=200,
                   help="mass histogram bins (default: %(default)s)")
    p.add_argument("--step-size", default="200 MB",
                   help="uproot.iterate chunk size (default: %(default)s)")
    p.add_argument("--max-events", type=int, default=None,
                   help="stop after this many events (quick tests)")
    p.add_argument("--no-gen-truth", action="store_true",
                   help="skip the GenJet truth method (e.g. an input with no "
                   "GenJet collection)")
    p.add_argument("--fit-nsigma", type=float, default=2.0, metavar="N",
                   help="Gaussian fit window, in current sigma either side of "
                   "the peak (default: %(default)s)")
    p.add_argument("--plot", default=None, metavar="PDF",
                   help="PDF to write (default: the --output path with a .pdf "
                   "suffix)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the PDF")
    p.add_argument("--plot-from", default=None, metavar="ROOT",
                   help="make the PDF from a previous run's output file and "
                   "exit - no models are run, so this works in an environment "
                   "that has matplotlib but not onnxruntime")
    p.add_argument("--write-tree", action="store_true",
                   help="also write a per-event TTree of every method's mass")
    args = p.parse_args()

    # Re-plot an existing run and stop: nothing below this needs to happen, and
    # requiring --config/inputs for it would defeat the purpose.
    if args.plot_from:
        pdf = Path(args.plot or Path(args.plot_from).with_suffix(".pdf"))
        pdf.parent.mkdir(parents=True, exist_ok=True)
        make_plots(entries_from_file(args.plot_from), pdf,
                   tuple(args.mass_range), title=Path(args.plot_from).stem)
        return

    if not args.inputs or not args.config:
        raise SystemExit(
            "Both an input and --config are required (or use --plot-from to "
            "re-plot a previous run)."
        )

    # Deferred so --plot-from never needs onnxruntime (see the note at the top).
    from run3_mj_evaluator.evaluate import (  # noqa: PLC0415
        _label_to_prefix,
        _load_session,
        candidate_fourvec,
        chunk_to_numpy,
        load_config,
        mass_asymmetry,
        prepare_comb_input,
        run_comb_solver,
        run_spanet,
    )

    if args.pool_size != 6:
        raise SystemExit(
            "--pool-size must be 6: only six jets split into two triplets with "
            "nothing left over, giving the 10 partitions this study enumerates. "
            "A larger pool needs an ISR choice too (7 jets -> 70 assignments, "
            "8 -> 280, as CombinatorialSolver enumerates them), which is a "
            "different baseline, not a bigger one."
        )

    jobs = resolve_inputs(args.inputs, args.tree)
    cfg = load_config(args.config)
    models = cfg["models"]

    print(f"inputs:  {len(jobs)} file(s)")
    print(f"config:  {args.config}")
    print(f"models:  {[m['label'] for m in models]}")
    print(f"pool:    leading {args.pool_size} jets for the Comb* baselines"
          + ("  |  restricted to exactly-6-jet events" if args.exactly_six else ""))
    print(f"match:   dR < {args.dr_max} to all jets\n")

    print("loading ONNX sessions...")
    sessions = []
    for m in models:
        t0 = time.perf_counter()
        disabled = ["MatMulAddFusion"] if m["type"] == "comb_solver" else None
        sessions.append(_load_session(m["path"], disabled_optimizers=disabled))
        print(f"  {m['label']} ({m['type']}) <- {m['path']} "
              f"[{time.perf_counter() - t0:.1f} s]")

    # Widest jet-slot count any consumer needs, so every chunk is padded to it
    # regardless of the multiplicities it happens to contain.
    min_width = max(
        [7, args.pool_size]
        + [int(m.get("num_jets", 8)) for m in models if m["type"] == "spanet"]
    )

    g1_table, g2_table = enumerate_partitions(args.pool_size)
    n_part = len(g1_table)
    print(f"\n{n_part} partitions of the leading {args.pool_size} jets\n")

    results = {}
    for m in models:
        pool = int(m.get("num_jets", 8)) if m["type"] == "spanet" else 7
        results[_label_to_prefix(m["label"])] = MethodResult(m["label"], pool)
    for i in range(n_part):
        results[f"Comb{i:02d}"] = MethodResult(f"Comb{i:02d}", args.pool_size)
    results["CombRandom"] = MethodResult("CombRandom", args.pool_size)
    results["TruthScoutingJet"] = MethodResult("TruthScoutingJet", None)
    if not args.no_gen_truth:
        results["TruthGenJet"] = MethodResult("TruthGenJet", None)

    rng = np.random.default_rng(args.seed)
    n_seen = n_used = n_truth = n_truth_gen = 0

    for path, tree in jobs:
        print(f"[read] {path}  (tree: {tree})", flush=True)
        for chunk in uproot.iterate(
            {path: tree}, filter_name=JET_BRANCHES + GENJET_BRANCHES + GEN_BRANCHES,
            step_size=args.step_size,
        ):
            n_seen += len(chunk)
            n_jets = ak.to_numpy(ak.num(chunk["ScoutingPFJet_pt"], axis=1))
            keep = n_jets == 6 if args.exactly_six else n_jets >= args.pool_size
            if args.max_events is not None:
                room = args.max_events - n_used
                if room <= 0:
                    break
                if keep.sum() > room:
                    allowed = np.cumsum(keep) <= room
                    keep = keep & allowed
            if not keep.any():
                continue
            chunk = chunk[keep]
            n_used += len(chunk)

            # Padded (N, J) kinematics in file order, plus the pT ordering.
            # min_jets forces the padded width up to what the widest consumer
            # needs: the comb_solver always wants 7 slots and a SPANet export
            # has its sequence length baked in, so a chunk of only 6-jet events
            # (which --exactly-six guarantees) would otherwise hand the ONNX
            # graph a too-narrow input and fail inside its attention reshape.
            pt, eta, phi, px, py, pz, e, mask = chunk_to_numpy(
                chunk, min_jets=min_width
            )
            n_chunk = len(pt)
            rows = np.arange(n_chunk)[:, None]
            order = np.argsort(np.where(mask, -pt, np.inf), axis=1)

            # Jets padded to a common width for the truth match: the nearest-jet
            # search must see every jet, not just the leading ones.
            width = pt.shape[1]
            eta_pad = np.full((n_chunk, MAX_JETS), np.nan)
            phi_pad = np.full((n_chunk, MAX_JETS), np.nan)
            valid_pad = np.zeros((n_chunk, MAX_JETS), dtype=bool)
            take = min(width, MAX_JETS)
            eta_pad[:, :take] = eta[:, :take]
            phi_pad[:, :take] = phi[:, :take]
            valid_pad[:, :take] = mask[:, :take]
            if width > MAX_JETS:
                print(f"  [warn] {width} jets in an event exceeds MAX_JETS="
                      f"{MAX_JETS}; the tail is excluded from truth matching")

            truth_idx, truth_ok, m_pole = truth_assignment(
                chunk, eta_pad, phi_pad, valid_pad, args.dr_max
            )
            n_truth += int(truth_ok.sum())

            gen_kin = (None if args.no_gen_truth
                       else collection_to_numpy(chunk, "GenJet", min_jets=6))

            pool = order[:, :args.pool_size]  # (N, pool) into the jet array

            # Cache the per-pool reachability, so each method is scored against
            # the ceiling its own jet pool imposes.
            reach_cache = {}

            def score(t1_idx, t2_idx, key, correct=None, kin=None, valid=None):
                """Reconstruct both tri-jets and hand the event to a method.

                ``kin`` overrides the jet collection the four-vectors are built
                from (the GenJet truth uses gen jets, everything else reco).
                ``valid`` NaNs out events where the method has no answer, so a
                method defined on a subset does not quietly fill zeros.
                """
                res = results[key]
                if res.pool not in reach_cache:
                    reach_cache[res.pool] = truth_in_pool(truth_idx, order, res.pool)
                k_pt, k_eta, k_phi, k_e = kin if kin is not None else (pt, eta, phi, e)
                _, _, _, m1 = candidate_fourvec(k_pt, k_eta, k_phi, k_e, t1_idx)
                _, _, _, m2 = candidate_fourvec(k_pt, k_eta, k_phi, k_e, t2_idx)
                m_avg = 0.5 * (m1 + m2)
                masym = mass_asymmetry(m1, m2)
                if valid is not None:
                    m_avg = np.where(valid, m_avg, np.nan)
                    masym = np.where(valid, masym, np.nan)
                res.add(
                    m_avg, m_pole, masym,
                    truth_ok, correct, reach_cache[res.pool] & truth_ok,
                )

            # --- the config's models, run exactly as evaluate.py runs them ---
            comb_norm, comb_raw, _s7, top7_idx = prepare_comb_input(
                pt, eta, phi, e, px, py, pz, mask
            )
            for model_cfg, session in zip(models, sessions):
                if model_cfg["type"] == "spanet":
                    n_src = int(model_cfg.get("num_jets", 8))
                    idx_n = order[:, :n_src]
                    rows_n = np.arange(n_chunk)[:, None]
                    ptn, etan, phin, en = (a[rows_n, idx_n] for a in (pt, eta, phi, e))
                    pxn, pyn, pzn = (a[rows_n, idx_n] for a in (px, py, pz))
                    maskn = mask[rows_n, idx_n]
                    ifmt = model_cfg["input_format"]
                    if ifmt == "cart":
                        source = np.stack([pxn, pyn, pzn, en], axis=-1)
                    elif ifmt == "spher_log":
                        source = np.stack(
                            [np.log1p(ptn), etan, phin, np.log1p(en)], axis=-1)
                    else:
                        source = np.stack([ptn, etan, phin, en], axis=-1)
                    t1_n, t2_n = run_spanet(session, source.astype(np.float32), maskn)
                    t1_idx = idx_n[rows_n, t1_n]
                    t2_idx = idx_n[rows_n, t2_n]
                else:
                    comb_in = comb_norm if model_cfg["normalized"] else comb_raw
                    t1_7, t2_7 = run_comb_solver(session, comb_in)
                    t1_idx = top7_idx[rows, t1_7]
                    t2_idx = top7_idx[rows, t2_7]
                correct = np.where(
                    truth_ok,
                    same_partition(t1_idx, t2_idx, truth_idx[:, 0], truth_idx[:, 1]),
                    False,
                )
                score(t1_idx, t2_idx, _label_to_prefix(model_cfg["label"]), correct)

            # --- all partitions of the leading pool, and a random one --------
            pick = rng.integers(0, n_part, size=n_chunk)
            rand_t1 = np.empty((n_chunk, 3), dtype=np.int64)
            rand_t2 = np.empty((n_chunk, 3), dtype=np.int64)
            for i in range(n_part):
                t1_idx = pool[rows, g1_table[i][None, :]]
                t2_idx = pool[rows, g2_table[i][None, :]]
                correct = np.where(
                    truth_ok,
                    same_partition(t1_idx, t2_idx, truth_idx[:, 0], truth_idx[:, 1]),
                    False,
                )
                score(t1_idx, t2_idx, f"Comb{i:02d}", correct)
                sel = pick == i
                rand_t1[sel] = t1_idx[sel]
                rand_t2[sel] = t2_idx[sel]

            correct_rand = np.where(
                truth_ok,
                same_partition(rand_t1, rand_t2, truth_idx[:, 0], truth_idx[:, 1]),
                False,
            )
            score(rand_t1, rand_t2, "CombRandom", correct_rand)

            # --- the two truth assignments: the resolution floors ------------
            # Reco jets under the true assignment: combinatorics removed, every
            # detector and HLT effect still present.
            safe_truth = np.where(truth_ok[:, None, None], truth_idx, 0)
            score(safe_truth[:, 0], safe_truth[:, 1], "TruthScoutingJet",
                  None, valid=truth_ok)

            # The same quarks matched to GenJets instead: no detector at all,
            # so what is left is jet clustering, out-of-cone radiation and
            # whatever the gluino's decay products lost to the underlying
            # event. The gap between the two is the reconstruction's share.
            if gen_kin is not None:
                g_pt, g_eta, g_phi, g_e, g_mask = gen_kin
                g_idx, g_ok, _ = truth_assignment(
                    chunk, g_eta, g_phi, g_mask, args.dr_max
                )
                n_truth_gen += int(g_ok.sum())
                safe_g = np.where(g_ok[:, None, None], g_idx, 0)
                score(safe_g[:, 0], safe_g[:, 1], "TruthGenJet",
                      None, kin=(g_pt, g_eta, g_phi, g_e), valid=g_ok)

        if args.max_events is not None and n_used >= args.max_events:
            break

    if n_used == 0:
        raise SystemExit("No events survived the jet-multiplicity selection.")

    for res in results.values():
        res.finalize()

    # --- report -----------------------------------------------------------
    print(f"\n{n_used:,} events used of {n_seen:,} read")
    print(f"truth-matched (ScoutingPFJet): {n_truth:,} "
          f"({100 * n_truth / n_used:.1f}%)")
    if not args.no_gen_truth:
        print(f"truth-matched (GenJet):       {n_truth_gen:,} "
              f"({100 * n_truth_gen / n_used:.1f}%)")

    # Fit the very histogram that gets written, so the table and any plot made
    # from the output file are describing the same binned distribution.
    mass_edges = np.linspace(args.mass_range[0], args.mass_range[1], args.bins + 1)
    for res in results.values():
        h = bh.Histogram(bh.axis.Regular(args.bins, *args.mass_range),
                         storage=bh.storage.Weight())
        clean = res.mass[np.isfinite(res.mass)]
        if len(clean):
            h.fill(clean)
        res.hist = h
        res.fit = gaussian_fit(
            h.view()["value"], mass_edges,
            float(np.median(clean)) if len(clean) else float("nan"),
            sigma_eff(res.mass), nsigma=args.fit_nsigma,
        )

    header = (f"\n{'method':<20}{'pool':>6}{'median m':>10}{'sigma_eff':>11}"
              f"{'sig/med':>9}{'fit m':>9}{'fit sig':>9}{'fit s/m':>9}"
              f"{'reach':>8}{'acc':>8}{'acc/reach':>11}"
              f"{'chance':>8}{'vs chance':>11}")
    print(header)
    print("-" * (len(header) - 1))
    for key, res in results.items():
        mass = res.mass[np.isfinite(res.mass)]
        if len(mass) == 0:
            print(f"{res.label:<20}{'(no events)':>10}")
            continue
        med = float(np.median(mass))
        sig = sigma_eff(mass)
        pool_s = "all" if res.pool is None else str(res.pool)
        reach = (100.0 * res.n_reachable / n_truth) if n_truth else float("nan")
        n_assign = n_assignments(res.pool)
        fit = res.fit
        if np.isfinite(fit["mu"]):
            # A fit that ran out of iterations is flagged, not hidden: its
            # window never settled, so the numbers are a snapshot of the last
            # pass rather than a converged core.
            flag = "" if fit["converged"] else "*"
            fit_m = f"{fit['mu']:.1f}{flag}"
            fit_s = f"{fit['sigma']:.1f}"
            fit_r = f"{fit['sigma'] / fit['mu']:.3f}"
        else:
            fit_m = fit_s = fit_r = "-"
        if key.startswith("Truth"):
            acc_s = ceil_s = chance_s = ratio_s = "-"
        else:
            acc_s = f"{100.0 * res.n_correct / n_truth:.1f}%" if n_truth else "-"
            on_reach = (res.n_correct / res.n_reachable) if res.n_reachable else None
            ceil_s = f"{100.0 * on_reach:.1f}%" if on_reach is not None else "-"
            chance_s = f"{100.0 / n_assign:.2f}%" if n_assign else "-"
            ratio_s = (f"{on_reach * n_assign:.1f}x"
                       if on_reach is not None and n_assign else "-")
        print(f"{res.label:<20}{pool_s:>6}{med:>10.1f}{sig:>11.1f}"
              f"{sig / med:>9.3f}{fit_m:>9}{fit_s:>9}{fit_r:>9}"
              f"{reach:>7.1f}%{acc_s:>8}"
              f"{ceil_s:>11}{chance_s:>8}{ratio_s:>11}")
    print("\nsigma_eff = half the central 68% interval, tails included. "
          "'fit m'/'fit sig' are\na Gaussian fitted iteratively over "
          f"+/-{args.fit_nsigma:g} sigma of the peak, so they describe the "
          "CORE\nonly and ignore the combinatorial shelf - expect fit sig < "
          "sigma_eff, and read\nthe two as answers to different questions. "
          "'*' marks a fit that did not converge.\n"
          "'pool' is how many leading jets the method chooses from; 'reach' is "
          "the share of\ntruth-matched events whose truth partition fits in "
          "that pool - its accuracy\nceiling. 'acc' is over all matched "
          "events, 'acc/reach' over the reachable ones.\n'chance' is "
          "1/(assignments the pool admits) and 'vs chance' is acc/reach over "
          "it -\nthe only column comparable across different pool sizes.")

    # --- histograms -------------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as out:
        for key, res in results.items():
            out[f"h_mass_{key}"] = res.hist  # the histogram the fit ran on
            mass_axis = bh.axis.Regular(args.bins, *args.mass_range)
            for name, values, axis in (
                (f"h_mass_matched_{key}", res.mass_matched, mass_axis),
                (f"h_resp_{key}", res.response, bh.axis.Regular(200, 0.0, 2.0)),
                (f"h_masym_{key}", res.masym, bh.axis.Regular(100, 0.0, 1.0)),
            ):
                h = bh.Histogram(axis, storage=bh.storage.Weight())
                clean = values[np.isfinite(values)]
                if len(clean):
                    h.fill(clean)
                out[name] = h
        summary = {
            key: {
                "label": res.label,
                "pool": res.pool,
                "n": int(np.isfinite(res.mass).sum()),
                "median_mass": float(np.median(res.mass[np.isfinite(res.mass)]))
                if np.isfinite(res.mass).any() else None,
                "sigma_eff": float(sigma_eff(res.mass)),
                "fit": res.fit,
                "n_correct": res.n_correct,
                "n_reachable": res.n_reachable,
            }
            for key, res in results.items()
        }
        out["summary"] = json.dumps(
            {"n_used": n_used, "n_truth_matched": n_truth,
             "n_truth_matched_genjet": n_truth_gen, "methods": summary}
        )
        if args.write_tree:
            branches = {f"mavg_{k}": np.float32 for k in results}
            out.mktree("resolution", branches)
            out["resolution"].extend(
                {f"mavg_{k}": res.mass.astype(np.float32)
                 for k, res in results.items()}
            )
    print(f"\nhistograms + summary written to {args.output}")

    if not args.no_plot:
        pdf = Path(args.plot or Path(args.output).with_suffix(".pdf"))
        pdf.parent.mkdir(parents=True, exist_ok=True)
        make_plots(entries_from_results(results, n_truth), pdf,
                   tuple(args.mass_range), title=Path(args.output).stem)


if __name__ == "__main__":
    main()
