#!/usr/bin/env python3
"""truth_matching_efficiency.py - how often the gen-truth jet assignment exists.

Every accuracy this repo quotes (``mass_resolution.py``, any model study) is
measured on truth-matched events, so the matcher's efficiency sets the sample
those numbers describe and the ceiling a model can reach. This script measures
it, on evaluator output or slimmed signal files (anything carrying
``ScoutingPFJet_*`` and the ``GenPart_*`` record).

What it reports
---------------
* **Efficiency vs the jets a model sees.** The quarks are matched against
  *all* jets (``truth_matching.truth_assignment``), then an event counts at
  ``N`` when all six matched jets are among the leading ``N`` by pT. That is
  the share of events a model with an ``N``-jet pool could possibly get right:
  N = 7 for the comb_solver models, 8 for SPANet. The ordering is the
  evaluator's own - ``argsort(-where(mask, pt, -inf))``, as in
  ``evaluate.prepare_comb_input`` and ``_topn`` - and *not* file order, which
  the slimmer's jet corrections leave unsorted (in 26% of RPV M1000 events, 37%
  of one QCD file). The table also shows what file order would have given.
* **A ΔR scan**: the same curve at several match radii (``--dr-scan``), plus
  the distribution of each quark's ΔR to its nearest jet, so the choice of
  0.4 can be judged against where the distribution actually falls.
* **Quark acceptance**: pT and \\|η\\| of the quarks with no jet inside
  ``--dr-max``, against all quarks, with the slimmer's jet cuts (pT > 20 GeV,
  \\|η\\| < 2.4) marked. A quark outside that acceptance cannot have a jet; one
  inside it that still has none was merged, lost or mis-measured.
* **Truth-jet pT rank**: for matched events, the pT rank of the lowest-ranked
  truth jet - i.e. the smallest ``N`` that contains the whole truth. Its
  cumulative sum is the reach curve, and its first bin answers how often the
  7th jet a comb_solver sees is extra radiation rather than a decay product.

The cutflow line separates the three things a match needs: a 2x3 gen record,
every quark within ΔR of a jet, and six *distinct* jets.

Usage
-----
    python scripts/truth_matching_efficiency.py \\
        ../example_root_files/slimmed_RPV_M1000_ScoutNano_RPV_M1000_1.root \\
        -o truth_eff_M1000.root

Inputs are ROOT files or a fileset/dataset JSON, as for ``mass_resolution.py``.
Mixing mass points is allowed but warned about: efficiency depends on the
boost. Output is a ROOT file of histograms with a JSON ``summary``, and a PDF
beside it; ``--plot-from truth_eff_M1000.root`` remakes the PDF from that file
alone.
"""

import argparse
import json
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
    JET_BRANCHES,
    collection_to_numpy,
    resolve_inputs,
)

#: Jet acceptance applied by the slimmer, drawn on the quark plots.
JET_PT_MIN = 20.0
JET_ABSETA_MAX = 2.4

#: Largest "lowest truth-jet rank" kept in its own bin; beyond is overflow.
MAX_RANK = 40

#: Reference categorical palette, slots 1-4 in fixed order (dataviz skill).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def dr_tag(dr):
    """0.4 -> 'dr0p40', a ROOT-safe key that survives a round trip."""
    return f"dr{dr:.2f}".replace(".", "p")


def evaluator_order(pt, mask):
    """Jet indices sorted the way the evaluator feeds its models (pT, descending)."""
    return np.argsort(-np.where(mask, pt, -np.inf), axis=1)


def lowest_rank(rank_of_jet, jets):
    """1-based pT rank of the lowest-ranked jet in ``jets``: the smallest N whose
    leading-N pool holds all of them."""
    rows = np.arange(len(jets))[:, None]
    return rank_of_jet[rows, jets].max(axis=1) + 1


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def run(args):
    spec = DECAY_SPECS[args.decay]
    scan = sorted(set(args.dr_scan) | {args.dr_max})
    jobs = resolve_inputs(args.inputs, args.tree)

    rank_axis = bh.axis.Integer(6, MAX_RANK + 1)
    h_rank = {dr: bh.Histogram(rank_axis) for dr in scan}
    h_rank_fileorder = bh.Histogram(rank_axis)
    h_dr = bh.Histogram(bh.axis.Regular(150, 0.0, 1.5))
    pt_axis = bh.axis.Regular(100, 0.0, 1000.0)
    eta_axis = bh.axis.Regular(50, 0.0, 5.0)
    h_pt = {k: bh.Histogram(pt_axis) for k in ("all", "unmatched")}
    h_eta = {k: bh.Histogram(eta_axis) for k in ("all", "unmatched")}

    n = {"used": 0, "balanced": 0, "within": 0, "matched": 0, "unsorted": 0,
         "ge7": 0, "ge7_7th_not_truth": 0}
    accept = {k: {"outside_eta": 0, "below_pt": 0, "in_acceptance": 0}
              for k in ("all", "unmatched")}
    pole_masses = set()

    for path, tree in jobs:
        print(f"[read] {path}  (tree: {tree})", flush=True)
        with uproot.open(path) as f:
            if tree not in f:
                print("  no events tree (cutflow-only file) - skipped")
                continue
            if "GenPart_pdgId" not in f[tree].keys():
                raise SystemExit(f"{path} has no GenPart record - truth matching "
                                 "needs MC signal (or ttbar with --decay ttbar).")
        for chunk in uproot.iterate({path: tree}, filter_name=JET_BRANCHES + GEN_BRANCHES,
                                    step_size=args.step_size):
            if args.max_events is not None:
                room = args.max_events - n["used"]
                if room <= 0:
                    break
                chunk = chunk[:room]
            n_ev = len(chunk)
            n["used"] += n_ev
            rows = np.arange(n_ev)[:, None]

            pt, eta, phi, _e, mask = collection_to_numpy(chunk, "ScoutingPFJet")
            n_jets = mask.sum(axis=1)
            order = evaluator_order(pt, mask)
            rank = np.empty_like(order)
            rank[rows, order] = np.arange(order.shape[1])[None, :]
            out_of_place = (order != np.arange(order.shape[1])[None, :]) & mask
            n["unsorted"] += int(out_of_place.any(axis=1).sum())

            q = truth_matching.match_quarks(chunk, eta, phi, mask, spec=spec)
            bal = q["balanced"]
            low = lowest_rank(rank, q["jet"])
            low_file = q["jet"].max(axis=1) + 1   # the same, taking jets in file order

            for dr in scan:
                ok = bal & truth_matching.within_distinct(q, dr)
                h_rank[dr].fill(low[ok])
                if dr == args.dr_max:
                    h_rank_fileorder.fill(low_file[ok])
                    n["matched"] += int(ok.sum())
                    n["balanced"] += int(bal.sum())
                    n["within"] += int((bal & (q["dr"] < dr).all(axis=1)).sum())
                    ge7 = ok & (n_jets >= 7)
                    n["ge7"] += int(ge7.sum())
                    n["ge7_7th_not_truth"] += int((ge7 & (low == 6)).sum())

            # Quark-level views use the gen-valid events only; elsewhere the
            # quark arrays are padding.
            q_dr = q["dr"][bal]
            q_pt = q["pt"][bal]
            q_abseta = np.abs(q["eta"][bal])
            h_dr.fill(np.where(np.isfinite(q_dr), q_dr, 99.0).ravel())
            unmatched = q_dr >= args.dr_max
            for key, sel in (("all", np.ones_like(unmatched)), ("unmatched", unmatched)):
                pts, etas = q_pt[sel], q_abseta[sel]
                h_pt[key].fill(pts)
                h_eta[key].fill(etas)
                out_eta = etas >= JET_ABSETA_MAX
                low_pt = ~out_eta & (pts < JET_PT_MIN)
                accept[key]["outside_eta"] += int(out_eta.sum())
                accept[key]["below_pt"] += int(low_pt.sum())
                accept[key]["in_acceptance"] += int((~out_eta & ~low_pt).sum())

            m_pole = truth_matching.resonance_mass(chunk, spec=spec)
            pole_masses.update(np.round(np.unique(m_pole[np.isfinite(m_pole)]), 1).tolist())

        if args.max_events is not None and n["used"] >= args.max_events:
            break

    if n["used"] == 0:
        raise SystemExit("No events read.")
    return {"spec": spec.label, "dr_max": args.dr_max, "scan": scan,
            "max_n": args.max_n, "counts": n, "acceptance": accept,
            "pole_masses": sorted(pole_masses)}, h_rank, h_rank_fileorder, h_dr, h_pt, h_eta


def eff_curve(rank_counts, n_used, max_n):
    """Fraction of events whose truth fits in the leading N, for N = 6..max_n,
    and the all-jets value. ``rank_counts`` are the Integer(6, ...) bin counts
    with the overflow bin last."""
    cum = np.cumsum(rank_counts)
    upto = [cum[min(k - 6, len(cum) - 1)] / n_used for k in range(6, max_n + 1)]
    return np.array(upto), cum[-1] / n_used


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(summary, rank_counts, fileorder_counts):
    c, max_n = summary["counts"], summary["max_n"]
    used, dr0 = c["used"], summary["dr_max"]
    pct = lambda k: 100.0 * k / used
    print(f"\n{used:,} events, truth: {summary['spec']}")
    masses = summary["pole_masses"]
    if not masses:
        print("[warn] no parent particle in the gen record")
    elif len(masses) > 1:
        print(f"[warn] several mass points mixed: {masses} - efficiency depends on "
              "the boost, so run them separately to compare")
    else:
        print(f"parent pole mass: {masses[0]:g} GeV")
    print(f"jets not in pT order in the file: {pct(c['unsorted']):.1f}% of events")

    print(f"\ncutflow at dR < {dr0:g} (of all events):")
    print(f"  gen record gives 2x3 quarks      {pct(c['balanced']):6.1f}%")
    print(f"  ... every quark within dR of a jet {pct(c['within']):6.1f}%")
    print(f"  ... on six distinct jets (matched) {pct(c['matched']):6.1f}%")

    cols = list(range(6, max_n + 1))
    head = f"{'dR':>6}" + "".join(f"{'N=' + str(k):>8}" for k in cols) + f"{'all':>8}"
    print("\nefficiency: truth matched AND all 6 truth jets among the leading N "
          "by pT (% of all events)")
    print(head)
    for dr in summary["scan"]:
        curve, total = eff_curve(rank_counts[dr], used, max_n)
        mark = " <-" if dr == dr0 else ""
        print(f"{dr:>6g}" + "".join(f"{100 * v:>8.1f}" for v in curve)
              + f"{100 * total:>8.1f}{mark}")
    curve, total = eff_curve(fileorder_counts, used, max_n)
    print(f"{'file':>6}" + "".join(f"{100 * v:>8.1f}" for v in curve)
          + f"{100 * total:>8.1f}   (dR < {dr0:g}, jets taken in FILE order - "
          "not what the models see)")

    m = c["matched"]
    if m:
        reach, _ = eff_curve(rank_counts[dr0], m, max_n)
        print(f"\nreach: share of MATCHED events whose truth fits in the leading N "
              f"(dR < {dr0:g})")
        print(" " * 6 + "".join(f"{'N=' + str(k):>8}" for k in cols))
        print(" " * 6 + "".join(f"{100 * v:>8.1f}" for v in reach))
        if c["ge7"]:
            print(f"matched events with >= 7 jets whose 7th jet is NOT a truth jet: "
                  f"{100 * c['ge7_7th_not_truth'] / c['ge7']:.1f}%")

    print(f"\nquark acceptance (gen-valid events; 'unmatched' = no jet within "
          f"dR < {dr0:g}):")
    print(f"{'':>12}{'|eta|>=2.4':>12}{'pT<20, in eta':>15}{'in acceptance':>15}{'quarks':>10}")
    for key, a in summary["acceptance"].items():
        tot = sum(a.values())
        if tot:
            print(f"{key:>12}{100 * a['outside_eta'] / tot:>11.1f}%"
                  f"{100 * a['below_pt'] / tot:>14.1f}%"
                  f"{100 * a['in_acceptance'] / tot:>14.1f}%{tot:>10,}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def load(root_path):
    """Everything the plots need, read back from an output file."""
    with uproot.open(root_path) as f:
        summary = json.loads(str(f["summary"]))
        rank = {dr: f[f"h_lowest_rank_{dr_tag(dr)}"].values(flow=True)[1:]
                for dr in summary["scan"]}
        get = lambda k: (f[k].values(), f[k].axis().edges())
        hists = {k: get(k) for k in ("h_quark_dr", "h_quark_pt_all", "h_quark_pt_unmatched",
                                     "h_quark_abseta_all", "h_quark_abseta_unmatched")}
    return summary, rank, hists


def make_plots(root_path, pdf_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print(f"[plot] matplotlib missing - no PDF. Remake it later with\n"
              f"       python scripts/truth_matching_efficiency.py --plot-from {root_path}")
        return
    summary, rank, hists = load(root_path)
    used, max_n, dr0 = summary["counts"]["used"], summary["max_n"], summary["dr_max"]
    masses = summary["pole_masses"]
    tag = f"m = {masses[0]:g} GeV" if len(masses) == 1 else "mixed mass points"
    ns = np.arange(6, max_n + 1)
    x_all = max_n + 1.5

    def style(ax):
        ax.grid(True, color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    with PdfPages(pdf_path) as pdf:
        # 1. efficiency vs N, one curve per match radius
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, dr in enumerate(summary["scan"]):
            curve, total = eff_curve(rank[dr], used, max_n)
            col, mk = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
            lw = 2.5 if dr == dr0 else 1.5
            ax.plot(ns, 100 * curve, color=col, marker=mk, markersize=7, linewidth=lw,
                    label=f"ΔR < {dr:g}" + (" (default)" if dr == dr0 else ""))
            ax.plot([x_all], [100 * total], color=col, marker=mk, markersize=7)
            ax.annotate(f"{100 * total:.0f}%", (x_all, 100 * total), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#52514e")
        for k, name in ((7, "comb_solver"), (8, "SPANet")):
            if k <= max_n:
                ax.axvline(k, color="#9a9994", linestyle=":", linewidth=1)
                ax.text(k, 1, f" {name}", rotation=90, va="bottom", fontsize=8, color="#52514e")
        ax.set_xticks(list(ns) + [x_all], [str(k) for k in ns] + ["all"])
        ax.set_xlabel("leading N jets by pT (the evaluator's order)")
        ax.set_ylabel("events truth-matched within the leading N [%]")
        ax.set_ylim(0, None)
        ax.set_title(f"Truth-matching efficiency vs jets considered ({tag})")
        ax.legend(frameon=False, loc="upper left")
        style(ax)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2. lowest truth-jet rank, and its cumulative (reach) curve
        counts = rank[dr0]
        matched = counts.sum()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        shown = counts[: max_n - 5]
        a1.bar(ns, 100 * shown / matched, width=0.8, color=SERIES[0])
        rest = counts[max_n - 5:].sum()
        a1.bar([x_all], [100 * rest / matched], width=0.8, color=SERIES[0], hatch="//",
               edgecolor="#fcfcfb")
        a1.set_xticks(list(ns) + [x_all], [str(k) for k in ns] + [f">{max_n}"])
        a1.set_xlabel("pT rank of the lowest-ranked truth jet")
        a1.set_ylabel("matched events [%]")
        a1.set_title(f"Where the 6th truth jet sits (ΔR < {dr0:g})", fontsize=10)
        style(a1)
        reach, _ = eff_curve(counts, matched, max_n)
        a2.plot(ns, 100 * reach, color=SERIES[0], marker="o", markersize=7, linewidth=2)
        for k, v in zip(ns, reach):
            if k in (6, 7, 8):
                a2.annotate(f"{100 * v:.0f}%", (k, 100 * v), xytext=(9, -3),
                            textcoords="offset points", ha="left", fontsize=8,
                            color="#52514e")
        a2.set_xticks(ns)
        a2.set_ylim(0, 105)
        a2.set_xlabel("leading N jets by pT")
        a2.set_ylabel("matched events with all truth in the leading N [%]")
        a2.set_title("Reach: share of matched events a pool of N can get right", fontsize=10)
        style(a2)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3. quark -> nearest jet ΔR
        vals, edges = hists["h_quark_dr"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.stairs(vals, edges, color=SERIES[0], linewidth=2)
        for dr in summary["scan"]:
            ax.axvline(dr, color="#9a9994", linestyle="-" if dr == dr0 else ":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("ΔR(quark, nearest jet)")
        ax.set_ylabel("quarks")
        ax.set_title(f"Quark-to-jet distance, gen-valid events ({tag}); lines = scanned radii",
                     fontsize=10)
        style(ax)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4. quark acceptance: all vs unmatched, unit area
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, var, cut, xlabel in ((axes[0], "pt", JET_PT_MIN, "quark pT [GeV]"),
                                     (axes[1], "abseta", JET_ABSETA_MAX, "quark |η|")):
            for i, key in enumerate(("all", "unmatched")):
                v, e = hists[f"h_quark_{var}_{key}"]
                if v.sum() > 0:
                    ax.stairs(v / v.sum(), e, color=SERIES[i], linewidth=2,
                              label="all quarks" if key == "all"
                              else f"no jet within ΔR < {dr0:g}")
            ax.axvline(cut, color="#52514e", linestyle="--", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("fraction of quarks")
            style(ax)
        axes[0].set_xlim(0, 600)
        axes[0].legend(frameon=False)
        axes[0].set_title("dashed: slimmer jet pT > 20 GeV", fontsize=10)
        axes[1].set_title("dashed: slimmer jet |η| < 2.4", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"plots written to {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Truth-matching efficiency vs the number "
                                "of leading jets, with a ΔR scan and quark acceptance.")
    p.add_argument("inputs", nargs="*", help="evaluator/slimmer ROOT file(s) or a JSON "
                   "listing them (not needed with --plot-from)")
    p.add_argument("-o", "--output", default="truth_matching_efficiency.root")
    p.add_argument("--tree", default=None, help="input tree (default: 'events' or the JSON's)")
    p.add_argument("--decay", default="rpv", choices=sorted(DECAY_SPECS))
    p.add_argument("--dr-max", type=float, default=0.4,
                   help="the match radius the headline numbers use (default: %(default)s)")
    p.add_argument("--dr-scan", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5],
                   help="radii for the scan (default: %(default)s)")
    p.add_argument("--max-n", type=int, default=12,
                   help="largest leading-N shown individually (default: %(default)s)")
    p.add_argument("--step-size", default="200 MB")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--plot", default=None, metavar="PDF")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-from", default=None, metavar="ROOT",
                   help="remake the PDF from a previous output file and exit")
    args = p.parse_args()

    if args.plot_from:
        make_plots(args.plot_from, Path(args.plot or Path(args.plot_from).with_suffix(".pdf")))
        return
    if not args.inputs:
        raise SystemExit("Give input file(s), or --plot-from a previous output.")
    if not 6 <= args.max_n <= MAX_RANK:
        raise SystemExit(f"--max-n must be between 6 and {MAX_RANK}.")

    summary, h_rank, h_rank_file, h_dr, h_pt, h_eta = run(args)
    if summary["counts"]["balanced"] == 0:
        raise SystemExit(f"No event has a {summary['spec']} decay in its gen record - "
                         "wrong sample (e.g. QCD), or the wrong --decay.")
    rank_counts = {dr: h.values(flow=True)[1:] for dr, h in h_rank.items()}
    report(summary, rank_counts, h_rank_file.values(flow=True)[1:])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as out:
        for dr, h in h_rank.items():
            out[f"h_lowest_rank_{dr_tag(dr)}"] = h
        out["h_lowest_rank_fileorder"] = h_rank_file
        out["h_quark_dr"] = h_dr
        for key in ("all", "unmatched"):
            out[f"h_quark_pt_{key}"] = h_pt[key]
            out[f"h_quark_abseta_{key}"] = h_eta[key]
        out["summary"] = json.dumps(summary)
    print(f"\nhistograms + summary written to {args.output}")

    if not args.no_plot:
        make_plots(args.output, Path(args.plot or Path(args.output).with_suffix(".pdf")))


if __name__ == "__main__":
    main()
