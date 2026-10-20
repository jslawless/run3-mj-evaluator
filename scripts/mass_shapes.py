#!/usr/bin/env python3
"""mass_shapes.py - signal mass shapes and pairing accuracy per model, as slides.

Reads **evaluator output** (``evaluated_*.root``): every model's pairing is
taken from the ``{Model}Candidate_*`` branches production wrote, so what is
studied is exactly what the analysis sees, with no ONNX run and no
onnxruntime. Models are discovered from the file (``*Candidate_jetIdx0``);
``--config`` (the evaluator config that made the file) supplies each model's
jet pool - 7 for comb_solver, ``num_jets`` for SPANet.

The observable is the average tri-jet mass ``(m1 + m2) / 2``, as in
``mass_resolution.py``. Each model is compared with three references, built
here from the same events:

* ``Truth (scouting jets)`` - the gen-truth pairing applied to the reco jets
  (``truth_matching.truth_assignment`` over all jets, ΔR < ``--dr-max``).
  Pairing errors gone, detector and HLT effects kept.
* ``Truth (gen jets)`` - the same quarks matched to GenJets, tri-jets built
  from those: no detector at all.
* ``Random (scouting jets)`` - one of the 10 splits of the leading 6 jets by
  pT, drawn per event (``--seed``): the no-information baseline.

Accuracy - the model's two triplets equal the truth's, ignoring order - is
quoted against three denominators:

* all events,
* events with a truth match using any number of jets,
* events whose six truth jets all sit inside the model's own pool (the
  leading 7 or 8 by pT): the events the model *could* have got right.

Slides (16:9 PDF pages)
-----------------------
1. Summary: unit-area shapes of every model and the three references.
2. One per model, in counts: left, the model's shape stacked into right /
   wrong (truth-matched events) / no truth, with the three references drawn
   over it; middle, the same after the analysis cut A < ``--masym-cut``
   (every curve using its own pairing's asymmetry); right, the numbers -
   the three accuracies with binomial errors, the Gaussian-core peak and
   width, sigma_eff (half the central 68%, tails included), and the share of
   events within +/-``--window`` of the gluino mass.

Usage
-----
    python scripts/mass_shapes.py evaluated_RPV_M1000_ScoutNano_RPV_M1000_1.root \\
        --config config/config.json -o mass_shapes_M1000.root

A ``root://cmseos.fnal.gov//store/...`` URL or a fileset JSON works as input.
Use one mass point per run. The ROOT file holds every histogram and a JSON
``summary``; ``--plot-from mass_shapes_M1000.root`` remakes the PDF from it
(e.g. with ``../run3-mj-analyzer/pkg-env/bin/python``, which has matplotlib).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from run3_mj_evaluator import truth_matching  # noqa: E402
from mass_resolution import (  # noqa: E402
    DECAY_SPECS,
    GEN_BRANCHES,
    GENJET_BRANCHES,
    JET_BRANCHES,
    collection_to_numpy,
    enumerate_partitions,
    gaussian_fit,
    resolve_inputs,
    same_partition,
    sigma_eff,
)
from truth_matching_efficiency import evaluator_order, lowest_rank  # noqa: E402

REFS = [("TruthScouting", "Truth (scouting jets)"),
        ("TruthGen", "Truth (gen jets)"),
        ("Random", "Random (scouting jets)")]
CATS = ("correct", "wrong", "notruth")
SELS = ("all", "acut")

#: Reference categorical palette, slots 1-5 in fixed order (dataviz skill);
#: a model keeps its slot on every slide.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948"]
REF_STYLE = {"TruthScouting": {"color": "#0b0b0b", "linestyle": "-"},
             "TruthGen": {"color": "#0b0b0b", "linestyle": "--"},
             "Random": {"color": "#8d8c87", "linestyle": ":"}}
CAT_STYLE = {"correct": ("right pairing", "#2a78d6"),
             "wrong": ("wrong pairing", "#eb6834"),
             "notruth": ("no truth match", "#d3d2cc")}


def label_to_prefix(label):
    """evaluate._label_to_prefix, without importing onnxruntime."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in label)


def model_pools(config_path):
    """{branch prefix: (label, jet pool)} from an evaluator config."""
    with open(config_path) as f:
        cfg = json.load(f)
    out = {}
    for m in cfg["models"]:
        pool = int(m.get("num_jets", 8)) if m["type"] == "spanet" else 7
        out[label_to_prefix(m["label"])] = (m["label"], pool)
    return out


def triplet_mass(pt, eta, phi, m, idx):
    """Invariant mass of the jets ``idx`` (N, 3) for padded (N, J) kinematics."""
    rows = np.arange(len(idx))[:, None]
    p, e_, f_, mm = (a[rows, idx] for a in (pt, eta, phi, m))
    px, py, pz = (p * np.cos(f_)).sum(1), (p * np.sin(f_)).sum(1), (p * np.sinh(e_)).sum(1)
    e = np.sqrt((p * np.cosh(e_)) ** 2 + mm ** 2).sum(1)
    return np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def masym(m1, m2):
    den = m1 + m2
    return np.where(den > 0, np.abs(m1 - m2) / np.where(den > 0, den, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def collect(args):
    """Per-event arrays for every model and reference, concatenated over files."""
    spec = DECAY_SPECS[args.decay]
    pools = model_pools(args.config)
    jobs = resolve_inputs(args.inputs, args.tree)
    g1_tab, g2_tab = enumerate_partitions(6)
    rng = np.random.default_rng(args.seed)

    models = None
    cols = {}
    def push(key, arr):
        cols.setdefault(key, []).append(np.asarray(arr))

    for path, tree in jobs:
        print(f"[read] {path}", flush=True)
        with uproot.open(path) as f:
            if tree not in f:
                print("  no events tree (cutflow-only file) - skipped")
                continue
            keys = set(f[tree].keys())
        found = sorted(m.group(1) for k in keys
                       if (m := re.match(r"^(.+?)Candidate_jetIdx0$", k)))
        found = [p for p in found if not p.endswith("Gen")]
        if models is None:
            models = [p for p in pools if p in found] + [p for p in found if p not in pools]
            missing = [p for p in found if p not in pools]
            if missing:
                print(f"[warn] {missing} not in {args.config}; assuming a 7-jet pool")
            print(f"models: {models}")
        elif set(found) != set(models):
            raise SystemExit(f"{path} has models {found}, earlier files {models}")
        has_gen = "GenJet_pt" in keys

        branches = JET_BRANCHES + GEN_BRANCHES + (GENJET_BRANCHES if has_gen else [])
        branches += [f"{p}Candidate_{v}" for p in models
                     for v in ("mass", "massAsymmetry", "jetIdx0", "jetIdx1", "jetIdx2")]
        for chunk in uproot.iterate({path: tree}, filter_name=branches,
                                    step_size=args.step_size):
            n_ev = len(chunk)
            rows = np.arange(n_ev)[:, None]
            pt, eta, phi, _e, mask = collection_to_numpy(chunk, "ScoutingPFJet", min_jets=7)
            mass_j = ak.to_numpy(ak.fill_none(ak.pad_none(
                chunk["ScoutingPFJet_m"], pt.shape[1], axis=1, clip=True), 0.0))
            order = evaluator_order(pt, mask)
            rank = np.empty_like(order)
            rank[rows, order] = np.arange(order.shape[1])[None, :]

            t_idx, ok = truth_matching.truth_assignment(chunk, eta, phi, mask, spec=spec,
                                                        dr_max=args.dr_max)
            low = np.where(ok, lowest_rank(rank, t_idx.reshape(n_ev, 6)), 99)
            push("truth_ok", ok)
            push("truth_low", low)
            push("pole", truth_matching.resonance_mass(chunk, spec=spec))

            safe = np.where(ok[:, None, None], t_idx, 0)
            m1 = triplet_mass(pt, eta, phi, mass_j, safe[:, 0])
            m2 = triplet_mass(pt, eta, phi, mass_j, safe[:, 1])
            push("TruthScouting_m", np.where(ok, 0.5 * (m1 + m2), np.nan))
            push("TruthScouting_A", np.where(ok, masym(m1, m2), np.nan))

            if has_gen and not args.no_gen_truth:
                g_pt, g_eta, g_phi, _ge, g_mask = collection_to_numpy(chunk, "GenJet", min_jets=6)
                g_m = ak.to_numpy(ak.fill_none(ak.pad_none(
                    chunk["GenJet_mass"], g_pt.shape[1], axis=1, clip=True), 0.0))
                g_idx, g_ok = truth_matching.truth_assignment(chunk, g_eta, g_phi, g_mask,
                                                              spec=spec, dr_max=args.dr_max)
                gs = np.where(g_ok[:, None, None], g_idx, 0)
                gm1 = triplet_mass(g_pt, g_eta, g_phi, g_m, gs[:, 0])
                gm2 = triplet_mass(g_pt, g_eta, g_phi, g_m, gs[:, 1])
                push("TruthGen_m", np.where(g_ok, 0.5 * (gm1 + gm2), np.nan))
                push("TruthGen_A", np.where(g_ok, masym(gm1, gm2), np.nan))
            else:
                push("TruthGen_m", np.full(n_ev, np.nan))
                push("TruthGen_A", np.full(n_ev, np.nan))

            lead6 = order[:, :6]
            pick = rng.integers(0, len(g1_tab), size=n_ev)
            r1 = lead6[rows, g1_tab[pick]]
            r2 = lead6[rows, g2_tab[pick]]
            rm1 = triplet_mass(pt, eta, phi, mass_j, r1)
            rm2 = triplet_mass(pt, eta, phi, mass_j, r2)
            push("Random_m", 0.5 * (rm1 + rm2))
            push("Random_A", masym(rm1, rm2))

            for p in models:
                mm = ak.to_numpy(chunk[f"{p}Candidate_mass"]).astype(np.float64)
                idx = np.stack([ak.to_numpy(chunk[f"{p}Candidate_jetIdx{k}"])
                                for k in range(3)], axis=-1)        # (N, 2, 3)
                right = ok & same_partition(idx[:, 0], idx[:, 1], t_idx[:, 0], t_idx[:, 1])
                push(f"{p}_m", mm.mean(axis=1))
                push(f"{p}_A", ak.to_numpy(chunk[f"{p}Candidate_massAsymmetry"]))
                push(f"{p}_right", right)

    if not cols:
        raise SystemExit("No events read.")
    data = {k: np.concatenate(v) for k, v in cols.items()}
    if not data["truth_ok"].any():
        raise SystemExit("No event truth-matched - is this signal, with the right --decay?")
    labels = {p: pools.get(p, (p, 7))[0] for p in models}
    pool = {p: pools.get(p, (p, 7))[1] for p in models}
    return data, models, labels, pool


# ---------------------------------------------------------------------------
# Numbers and histograms
# ---------------------------------------------------------------------------

def frac(k, n):
    """(value, binomial error) of k/n."""
    if n == 0:
        return None, None
    p = k / n
    return p, float(np.sqrt(p * (1 - p) / n))


def shape_numbers(m, pole, window, edges, nsigma):
    clean = m[np.isfinite(m)]
    if len(clean) == 0:
        return {"n": 0}
    counts, _ = np.histogram(clean, edges)
    fit = gaussian_fit(counts.astype(float), edges, float(np.median(clean)),
                       sigma_eff(clean), nsigma=nsigma)
    in_win = np.abs(clean - pole) < window * pole if np.isfinite(pole) else np.zeros(0)
    return {"n": int(len(clean)), "median": float(np.median(clean)),
            "sigma_eff": float(sigma_eff(clean)), "fit": fit,
            "in_window": float(in_win.mean()) if len(in_win) else None}


def build(data, models, labels, pool, args):
    poles = data["pole"][np.isfinite(data["pole"])]
    uniq = np.unique(np.round(poles, 1))
    if len(uniq) > 1:
        print(f"[warn] mass points mixed: {uniq.tolist()} - run them separately")
    pole = float(np.median(poles)) if len(poles) else float("nan")
    lo, hi = args.mass_range or (0.0, 2.0 * pole if np.isfinite(pole) else 2000.0)
    edges = np.linspace(lo, hi, args.bins + 1)
    n_all = len(data["truth_ok"])
    ok = data["truth_ok"]

    hists, summary = {}, {"pole": pole, "n_events": n_all, "n_truth": int(ok.sum()),
                          "masym_cut": args.masym_cut, "window": args.window,
                          "dr_max": args.dr_max, "edges": [lo, hi, args.bins],
                          "models": models, "labels": labels, "pool": pool, "stats": {}}
    for sel in SELS:
        for key, _lab in REFS:
            m, A = data[f"{key}_m"], data[f"{key}_A"]
            keep = np.isfinite(m) & ((A < args.masym_cut) if sel == "acut" else True)
            hists[f"h_{key}_all_{sel}"] = np.histogram(m[keep], edges)[0]
            summary["stats"][f"{key}_{sel}"] = shape_numbers(
                m[keep], pole, args.window, edges, args.fit_nsigma)
        for p in models:
            m, A, right = data[f"{p}_m"], data[f"{p}_A"], data[f"{p}_right"]
            keep = (A < args.masym_cut) if sel == "acut" else np.ones(n_all, bool)
            reach = ok & (data["truth_low"] <= pool[p])
            cat = {"correct": right, "wrong": ok & ~right, "notruth": ~ok}
            for c in CATS:
                hists[f"h_{p}_{c}_{sel}"] = np.histogram(m[keep & cat[c]], edges)[0]
            k = int((right & keep).sum())
            st = shape_numbers(m[keep], pole, args.window, edges, args.fit_nsigma)
            st.update({
                "n_right": k, "n_sel": int(keep.sum()),
                "n_matched": int((ok & keep).sum()), "n_reach": int((reach & keep).sum()),
                "acc_all": frac(k, int(keep.sum())),
                "acc_matched": frac(k, int((ok & keep).sum())),
                "acc_reach": frac(k, int((reach & keep).sum())),
                "cut_eff": float(keep.mean()),
            })
            summary["stats"][f"{p}_{sel}"] = st
    return hists, summary, edges


def write(path, hists, summary, edges):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as out:
        for k, v in hists.items():
            out[k] = (v.astype(np.float64), edges)
        out["summary"] = json.dumps(summary)
    print(f"\nhistograms + summary written to {path}")


def report(summary):
    print(f"\n{summary['n_events']:,} events, gluino mass {summary['pole']:g} GeV, "
          f"truth-matched {100 * summary['n_truth'] / summary['n_events']:.1f}%")
    for sel, name in (("all", "all events"), ("acut", f"A < {summary['masym_cut']:g}")):
        print(f"\n[{name}]")
        print(f"{'model':<18}{'pool':>5}{'kept':>7}{'acc/all':>9}{'acc/match':>11}"
              f"{'acc/reach':>11}{'fit m':>8}{'fit sig':>9}{'sig_eff':>9}"
              f"{'in ±' + format(100 * summary['window'], '.0f') + '%':>8}")
        for p in summary["models"]:
            s = summary["stats"][f"{p}_{sel}"]
            fmt = lambda a: f"{100 * a[0]:.1f}%" if a[0] is not None else "-"
            fit = s.get("fit", {})
            fm = f"{fit['mu']:.0f}" if fit and np.isfinite(fit.get("mu", np.nan)) else "-"
            fs = f"{fit['sigma']:.0f}" if fit and np.isfinite(fit.get("sigma", np.nan)) else "-"
            print(f"{summary['labels'][p]:<18}{summary['pool'][p]:>5}"
                  f"{100 * s['cut_eff']:>6.1f}%{fmt(s['acc_all']):>9}{fmt(s['acc_matched']):>11}"
                  f"{fmt(s['acc_reach']):>11}{fm:>8}{fs:>9}"
                  f"{s.get('sigma_eff', float('nan')):>9.0f}"
                  f"{100 * (s.get('in_window') or 0):>7.1f}%")


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def load(root_path):
    with uproot.open(root_path) as f:
        summary = json.loads(str(f["summary"]))
        hists = {k.split(";")[0]: f[k].values() for k in f.keys() if k.startswith("h_")}
    lo, hi, nb = summary["edges"]
    return hists, summary, np.linspace(lo, hi, int(nb) + 1)


def make_slides(root_path, pdf_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print(f"[plot] matplotlib missing - no PDF. Remake it later with\n"
              f"       python scripts/mass_shapes.py --plot-from {root_path}")
        return
    hists, s, edges = load(root_path)
    models, labels = s["models"], s["labels"]
    pole, cut = s["pole"], s["masym_cut"]
    centres_w = np.diff(edges)
    ink, ink2 = "#0b0b0b", "#52514e"

    def style(ax):
        ax.grid(True, color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(pole, color="#b5b4ae", linewidth=1, zorder=0)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("average tri-jet mass (m₁ + m₂)/2 [GeV]")

    def unit(v):
        tot = v.sum()
        return v / (tot * centres_w) if tot > 0 else v

    def model_total(p, sel):
        return sum(hists[f"h_{p}_{c}_{sel}"] for c in CATS)

    pct = lambda a: "–" if a[0] is None else f"{100 * a[0]:.1f} ± {100 * a[1]:.1f}%"

    with PdfPages(pdf_path) as pdf:
        # --- summary slide --------------------------------------------------
        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        for i, p in enumerate(models):
            ax.stairs(unit(model_total(p, "all")), edges, color=SERIES[i % len(SERIES)],
                      linewidth=2.2, label=labels[p])
        for key, lab in REFS:
            v = hists[f"h_{key}_all_all"]
            if v.sum() > 0:
                ax.stairs(unit(v), edges, linewidth=2.2, label=lab, **REF_STYLE[key])
        style(ax)
        ax.set_ylabel("a.u. (unit area)")
        ax.set_title(f"Signal mass shapes by pairing method — m(g̃) = {pole:g} GeV, "
                     f"{s['n_events']:,} events", fontsize=15, loc="left")
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        pdf.savefig(fig)
        plt.close(fig)

        # --- one slide per model ----------------------------------------------
        for i, p in enumerate(models):
            fig = plt.figure(figsize=(13.33, 7.5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.25,
                                  left=0.06, right=0.97, top=0.86, bottom=0.24)
            fig.suptitle(f"{labels[p]} — pool: leading {s['pool'][p]} jets by pT, "
                         f"m(g̃) = {pole:g} GeV", x=0.06, ha="left", fontsize=15)
            ymax = 0
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
            for ax, sel, ttl in ((axes[0], "all", "all events"),
                                 (axes[1], "acut", f"A < {cut:g} (each method's own pairing)")):
                bottom = np.zeros(len(edges) - 1)
                for c in CATS:
                    v = hists[f"h_{p}_{c}_{sel}"]
                    name, col = CAT_STYLE[c]
                    ax.stairs(bottom + v, edges, baseline=bottom, fill=True, color=col,
                              label=f"{labels[p]}: {name}")
                    bottom = bottom + v
                for key, lab in REFS:
                    v = hists[f"h_{key}_all_{sel}"]
                    if v.sum() > 0:
                        ax.stairs(v, edges, linewidth=2, label=lab, **REF_STYLE[key])
                        ymax = max(ymax, v.max())
                ymax = max(ymax, bottom.max())
                style(ax)
                ax.set_title(ttl, fontsize=11, loc="left", color=ink2)
                ax.set_ylabel("events")
            for ax in axes:
                ax.set_ylim(0, 1.08 * ymax)
            handles, names = axes[0].get_legend_handles_labels()
            fig.legend(handles, names, frameon=False, fontsize=10, ncol=3,
                       loc="lower left", bbox_to_anchor=(0.055, 0.01))

            # numbers panel
            tx = fig.add_subplot(gs[0, 2])
            tx.axis("off")
            a, c_ = s["stats"][f"{p}_all"], s["stats"][f"{p}_acut"]
            rows = [("", "all events", f"A < {cut:g}"),
                    ("events kept", f"{a['n_sel']:,}", f"{c_['n_sel']:,} ({100 * c_['cut_eff']:.0f}%)"),
                    ("Accuracy (right pairings)", "", ""),
                    ("  / all events", pct(a["acc_all"]), pct(c_["acc_all"])),
                    ("  / truth-matched", pct(a["acc_matched"]), pct(c_["acc_matched"])),
                    (f"  / truth in top {s['pool'][p]}", pct(a["acc_reach"]), pct(c_["acc_reach"])),
                    ("  right pairings", f"{a['n_right']:,}", f"{c_['n_right']:,}"),
                    ("Shape", "", ""),
                    ("  fit peak [GeV]", *(f"{x['fit']['mu']:.0f}" if x.get("fit") and
                                           np.isfinite(x["fit"].get("mu", np.nan)) else "–"
                                           for x in (a, c_))),
                    ("  core σ [GeV]", *(f"{x['fit']['sigma']:.0f} ({100 * x['fit']['sigma'] / x['fit']['mu']:.0f}%)"
                                         if x.get("fit") and np.isfinite(x["fit"].get("sigma", np.nan))
                                         else "–" for x in (a, c_))),
                    ("  σ_eff (68%) [GeV]", *(f"{x['sigma_eff']:.0f}" if x.get("n") else "–"
                                              for x in (a, c_))),
                    (f"  in ±{100 * s['window']:.0f}% of m(g̃)",
                     *(f"{100 * x['in_window']:.1f}%" if x.get("in_window") is not None else "–"
                       for x in (a, c_)))]
            y = 0.98
            for r in rows:
                head = r[1] == "" and r[2] == "" and r[0]
                weight = "bold" if head or r[0] == "" else "normal"
                tx.text(0.0, y, r[0], fontsize=10.5, weight=weight, color=ink,
                        transform=tx.transAxes, va="top")
                tx.text(0.69, y, r[1], fontsize=10.5, weight=weight if r[0] == "" else "normal",
                        color=ink, transform=tx.transAxes, va="top", ha="right")
                tx.text(1.0, y, r[2], fontsize=10.5, weight=weight if r[0] == "" else "normal",
                        color=ink, transform=tx.transAxes, va="top", ha="right")
                y -= 0.062 if not head else 0.07
            tx.text(0.0, -0.02, f"Truth-matched (ΔR < {s['dr_max']:g}, any jets): "
                    f"{100 * s['n_truth'] / s['n_events']:.1f}% of events.\n"
                    f"'Truth in top {s['pool'][p]}': all six truth jets are among the\n"
                    f"leading {s['pool'][p]} by pT - the events this model could get right.\n"
                    "Core σ: Gaussian fit to the peak. σ_eff: half the central 68%,\n"
                    "wrong-pairing tail included.",
                    fontsize=9, color=ink2, transform=tx.transAxes, va="top")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"slides written to {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Signal mass shapes and pairing accuracy "
                                "per model, from evaluator output, as 16:9 slides.")
    p.add_argument("inputs", nargs="*", help="evaluated ROOT file(s) or a JSON listing them")
    p.add_argument("--config", default="config/config.json",
                   help="evaluator config that produced the files (for each model's "
                   "jet pool; default: %(default)s)")
    p.add_argument("-o", "--output", default="mass_shapes.root")
    p.add_argument("--tree", default=None)
    p.add_argument("--decay", default="rpv", choices=sorted(DECAY_SPECS))
    p.add_argument("--dr-max", type=float, default=0.4)
    p.add_argument("--masym-cut", type=float, default=0.1,
                   help="mass-asymmetry cut for the second panel (default: %(default)s)")
    p.add_argument("--window", type=float, default=0.2,
                   help="half-width of the mass window, as a fraction of m(gluino) "
                   "(default: %(default)s)")
    p.add_argument("--mass-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="histogram range in GeV (default: 0 to 2 x the gluino mass)")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--fit-nsigma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gen-truth", action="store_true")
    p.add_argument("--step-size", default="200 MB")
    p.add_argument("--plot", default=None, metavar="PDF")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-from", default=None, metavar="ROOT")
    args = p.parse_args()

    if args.plot_from:
        make_slides(args.plot_from, Path(args.plot or Path(args.plot_from).with_suffix(".pdf")))
        return
    if not args.inputs:
        raise SystemExit("Give evaluated input file(s), or --plot-from a previous output.")

    data, models, labels, pool = collect(args)
    hists, summary, edges = build(data, models, labels, pool, args)
    report(summary)
    write(args.output, hists, summary, edges)
    if not args.no_plot:
        make_slides(args.output, Path(args.plot or Path(args.output).with_suffix(".pdf")))


if __name__ == "__main__":
    main()
