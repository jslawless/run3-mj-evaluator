"""truth_matching.py - gen-truth jet assignment for pair-produced resonances.

One matcher for every topology in this analysis. Given a gen record and a jet
collection it answers: *which jets did each parent's decay quarks become*, as
jet indices that can be compared directly against a model's assignment.

Why one function covers ttbar and the RPV gluino
------------------------------------------------
The obvious way to split the six hard quarks into two groups is by charge -
``b`` from ``t`` vs ``b̄`` from ``t̄``, ``q q'`` from ``W+`` vs ``W-``. That is
what ``run3-mj-analyzer``'s ttbar-specific matcher does, and it **cannot** work
for the RPV signal: the gluino is a Majorana fermion, so both of them carry
pdgId ``+1000021`` and a sign test puts all six quarks in one group.

So the split here is structural instead of charge-based: walk up each quark's
mother chain to the nearest ancestor that is one of ``spec.parent_pdgids``, and
group by **which ancestor particle** it lands on. That handles

* ``go -> u d s`` (one step: quark -> gluino), and
* ``t -> b W, W -> q q'`` (two steps for the light quarks: quark -> W -> top),

with the same code and no special cases, because the walk does not care how
many intermediate resonances it passes on the way. It also works for
same-pdgId parents, which is exactly where the charge method fails.

Copies are collapsed. Pythia writes a resonance many times over as it showers
(the gluino appears with status 22, then 44, 52, 62 before decaying), and
different daughters can point at different copies of the same particle.
Grouping on the raw mother index would then split one decay in two, so each
ancestor is reduced to its **earliest copy** - walk further up while the mother
carries the same \\|pdgId\\| - which is a stable identifier for "this parent".

Matching
--------
Each quark takes its nearest jet by ΔR. An event is accepted only when all
``n_parents * n_prongs`` quarks are within ``dr_max`` of **distinct** jets.

That distinctness cut, not the gen record, is what limits the matched sample.
Measured on 10,724 RPV M1000 events: 60.7% of events have all six quarks within
ΔR<0.4 of some jet, but only 46.6% land on six different jets. The matching is
greedy - each quark picks independently and a collision rejects the event
rather than being re-optimised - so a global (Hungarian) assignment on the
ΔR matrix would recover a good part of that 14-point gap. That is the main
known improvement available here.

Other deliberate choices, all of which cost or buy efficiency:

* Quarks are taken at ``status == 23``, the hard-process copy, so a quark that
  radiates hard has its direction read before the radiation.
* No jet pT or eta cut is applied at matching time - whatever the caller passes
  in is used. The slimmer has already applied pT>20, \\|eta\\|<2.4.
* No reciprocity: a jet may be nearest to a quark without that quark being the
  nearest to the jet.
* ``dr_max`` defaults to the AK4 jet radius, which is conventional rather than
  optimised.

Index convention
----------------
The returned indices point into the jet arrays **exactly as the caller passed
them**, so pass jets in the same order the thing you are comparing against uses
(for evaluator models, that is file order - their pT-sorted pools are mapped
back before their indices are returned). Sorting the jets here and not there,
or the other way round, would silently compare two different index spaces.
"""

from dataclasses import dataclass

import awkward as ak
import numpy as np

__all__ = [
    "DecaySpec",
    "RPV_GLUINO",
    "TTBAR",
    "match_quarks",
    "within_distinct",
    "truth_assignment",
    "resonance_mass",
]

#: Pythia status of the outgoing partons of the hardest subprocess. ScoutingNano
#: stores ``GenPart_status`` but not NanoAOD's ``statusFlags`` bitfield, so this
#: is the available handle on "the hard quarks" - there is no ``isLastCopy``.
HARD_STATUS = 23

#: Quarks a resonance may decay to, by absolute pdgId (d, u, s, c, b).
QUARK_ABS_PDGIDS = (1, 2, 3, 4, 5)

#: Safety stop for the mother-chain walks. Gen records are shallow; this only
#: exists so a malformed record cannot spin forever.
MAX_WALK = 64


@dataclass(frozen=True)
class DecaySpec:
    """Which resonance decays to how many quarks, pair-produced.

    parent_pdgids
        Absolute pdgIds that count as "the parent". The walk stops at the first
        ancestor matching any of them.
    n_prongs
        Quarks per parent (3 for both ``go -> qqq`` and ``t -> b q q'``).
    n_parents
        How many parents per event (2 - these are pair-production topologies).
    label
        Human-readable, used only in messages.
    """

    parent_pdgids: tuple
    n_prongs: int = 3
    n_parents: int = 2
    label: str = ""

    @property
    def n_quarks(self):
        return self.n_parents * self.n_prongs


#: RPV gluino pair, each to three quarks via lambda''_112. One step from quark
#: to parent, and both parents share pdgId +1000021 (Majorana).
RPV_GLUINO = DecaySpec((1000021,), 3, 2, "go -> q q q")

#: All-hadronic ttbar. Two steps for the light quarks (q -> W -> t), one for
#: the b. The parent is the top; the W is passed through by the walk.
TTBAR = DecaySpec((6,), 3, 2, "t -> b W(-> q q')")


def _gen_arrays(chunk, prefix, fields):
    """Padded ``(N, P)`` numpy arrays for the requested ``<prefix>_<field>``.

    Padding values are chosen to be inert: pdgId 0 matches no selection and
    mother -1 terminates a walk.
    """
    fills = {"genPartIdxMother": -1}
    counts = ak.num(chunk[f"{prefix}_{fields[0]}"], axis=1)
    width = int(ak.max(counts)) if len(counts) else 0
    width = max(width, 1)
    out = []
    for field in fields:
        arr = chunk[f"{prefix}_{field}"]
        padded = ak.fill_none(
            ak.pad_none(arr, width, axis=1, clip=True), fills.get(field, 0)
        )
        out.append(ak.to_numpy(padded))
    return out


def _walk_to_parent(pdg, mother, parent_pdgids, max_walk=MAX_WALK):
    """Index of each particle's nearest ancestor whose \\|pdgId\\| is a parent.

    Returns ``(N, P)`` int, -1 where the chain reaches the beam without
    passing one. Starts at the particle's mother, so a parent is never its own
    ancestor.
    """
    n_events, n_part = pdg.shape
    rows = np.arange(n_events)[:, None]
    is_parent = np.isin(np.abs(pdg), np.asarray(parent_pdgids))

    found = np.full((n_events, n_part), -1, dtype=np.int64)
    cur = mother.astype(np.int64)
    active = cur >= 0
    for _ in range(max_walk):
        if not active.any():
            break
        safe = np.where(active, cur, 0)
        hit = active & is_parent[rows, safe]
        found = np.where(hit & (found < 0), safe, found)
        nxt = mother[rows, safe].astype(np.int64)
        active = active & ~hit & (nxt >= 0)
        cur = np.where(active, nxt, cur)
    return found


def _earliest_copy(idx, pdg, mother, max_walk=MAX_WALK):
    """Reduce each index to the earliest copy of that same particle.

    Walks up while the mother carries the same \\|pdgId\\|, which collapses a
    Pythia shower chain (status 22 -> 44 -> 52 -> 62) onto one representative,
    so two daughters pointing at different copies still group together.
    """
    n_events, _ = pdg.shape
    rows = np.arange(n_events)[:, None]
    out = idx.copy()
    active = out >= 0
    for _ in range(max_walk):
        if not active.any():
            break
        safe = np.where(active, out, 0)
        mum = mother[rows, safe].astype(np.int64)
        mum_safe = np.where(mum >= 0, mum, 0)
        same = (
            active
            & (mum >= 0)
            & (np.abs(pdg[rows, mum_safe]) == np.abs(pdg[rows, safe]))
        )
        out = np.where(same, mum, out)
        active = same
    return out


def match_quarks(
    chunk,
    jet_eta,
    jet_phi,
    jet_valid,
    spec=RPV_GLUINO,
    hard_status=HARD_STATUS,
    prefix="GenPart",
):
    """Per-quark detail behind ``truth_assignment``, before any ``dr_max`` cut.

    Selects the hard decay quarks, groups them by parent, and finds each one's
    nearest jet. Nothing here depends on the match radius, so a caller scanning
    ``dr_max`` (or studying the quarks that fail it) can run this once and cut
    afterwards; ``truth_assignment`` is exactly this plus the radius and
    distinct-jet requirements.

    Parameters are as for ``truth_assignment``.

    Returns
    -------
    dict of
        pt, eta : (N, n_quarks) float
            Quark kinematics, in parent-grouped order: quarks ``[0:n_prongs]``
            share one parent, the next ``n_prongs`` the next, and so on.
        jet : (N, n_quarks) int
            Index of each quark's nearest valid jet (meaningless if none).
        dr : (N, n_quarks) float
            ΔR to that jet; ``inf`` when the event has no valid jet.
        balanced : (N,) bool
            The gen record yielded exactly ``n_parents`` groups of
            ``n_prongs`` quarks. Where False the other arrays are padding.
    """
    pdg, status, mother, eta, phi = _gen_arrays(
        chunk, prefix, ("pdgId", "status", "genPartIdxMother", "eta", "phi")
    )
    # pT is reported, never used for matching, so a record without it still matches.
    if f"{prefix}_pt" in ak.fields(chunk):
        (pt,) = _gen_arrays(chunk, prefix, ("pt",))
    else:
        pt = np.full(pdg.shape, np.nan)
    n_events = pdg.shape[0]
    n_quarks, n_prongs, n_parents = spec.n_quarks, spec.n_prongs, spec.n_parents

    ancestor = _earliest_copy(
        _walk_to_parent(pdg, mother, spec.parent_pdgids), pdg, mother
    )

    is_quark = np.isin(np.abs(pdg), np.asarray(QUARK_ABS_PDGIDS))
    selected = (status == hard_status) & is_quark & (ancestor >= 0)

    # Gather the selected quarks into a fixed (N, n_quarks) block. Events with
    # the wrong number are rejected via `balanced` below; the gather is padded
    # so the array shape is static either way.
    n_found = selected.sum(axis=1)
    order = np.argsort(~selected, axis=1, kind="stable")  # selected first
    take = order[:, :n_quarks]
    rows = np.arange(n_events)[:, None]
    q_pt = pt[rows, take]
    q_eta = eta[rows, take]
    q_phi = phi[rows, take]
    q_anc = np.where(selected[rows, take], ancestor[rows, take], -1)

    # Nearest jet per quark: (N, n_quarks, J) -> (N, n_quarks).
    dr = np.hypot(
        q_eta[:, :, None] - jet_eta[:, None, :],
        _wrap_phi(q_phi[:, :, None] - jet_phi[:, None, :]),
    )
    dr = np.where(jet_valid[:, None, :], dr, np.inf)
    nearest = dr.argmin(axis=2)
    nearest_dr = dr.min(axis=2)

    # Sort each event's quarks by ancestor index, so a parent's prongs land in
    # one contiguous block and the parent ordering is reproducible.
    anc_order = np.argsort(q_anc, axis=1, kind="stable")
    anc_sorted = np.take_along_axis(q_anc, anc_order, axis=1)

    def by_parent(a):
        return np.take_along_axis(a, anc_order, axis=1)

    blocks = anc_sorted.reshape(n_events, n_parents, n_prongs)
    # Every prong in a block shares one ancestor, and the blocks differ.
    same_within = (blocks == blocks[:, :, :1]).all(axis=2).all(axis=1)
    distinct_blocks = np.ones(n_events, dtype=bool)
    for b in range(1, n_parents):
        distinct_blocks &= blocks[:, b, 0] != blocks[:, b - 1, 0]

    balanced = (n_found == n_quarks) & same_within & distinct_blocks & (
        anc_sorted >= 0
    ).all(axis=1)

    return {
        "pt": by_parent(q_pt),
        "eta": by_parent(q_eta),
        "jet": by_parent(nearest),
        "dr": by_parent(nearest_dr),
        "balanced": balanced,
    }


def truth_assignment(
    chunk,
    jet_eta,
    jet_phi,
    jet_valid,
    spec=RPV_GLUINO,
    dr_max=0.4,
    hard_status=HARD_STATUS,
    prefix="GenPart",
):
    """Match a pair of resonances' decay quarks to jets.

    Parameters
    ----------
    chunk : awkward.Array
        One chunk carrying flat ``<prefix>_*`` gen branches.
    jet_eta, jet_phi : (N, J) float
        Padded jet directions, in whatever order the caller will use the
        returned indices in (see the module docstring).
    jet_valid : (N, J) bool
        True where the padded slot holds a real jet.
    spec : DecaySpec
        The topology. ``RPV_GLUINO`` or ``TTBAR``.
    dr_max : float
        Match radius.
    hard_status : int
        Gen status of the hard decay quarks.
    prefix : str
        Gen-particle branch prefix.

    Returns
    -------
    idx : (N, n_parents, n_prongs) int
        Jet indices per parent, parents ordered by their gen-record index so
        the result is deterministic. Meaningless where ``ok`` is False.
    ok : (N,) bool
        Every quark matched, within ``dr_max``, to a distinct jet.
    """
    q = match_quarks(chunk, jet_eta, jet_phi, jet_valid, spec=spec,
                     hard_status=hard_status, prefix=prefix)
    ok = q["balanced"] & within_distinct(q, dr_max)
    idx = q["jet"].reshape(len(ok), spec.n_parents, spec.n_prongs)
    return idx, ok


def within_distinct(q, dr_max):
    """Per event: every quark within ``dr_max`` of its jet, all jets distinct.

    ``q`` is a ``match_quarks`` result; combine with ``q["balanced"]`` to get
    ``truth_assignment``'s ``ok``.
    """
    within = (q["dr"] < dr_max).all(axis=1)
    ordered = np.sort(q["jet"], axis=1)
    distinct_jets = (np.diff(ordered, axis=1) > 0).all(axis=1)
    return within & distinct_jets


def resonance_mass(chunk, spec=RPV_GLUINO, prefix="GenPart"):
    """Parent mass from the gen record, per event; NaN where none is present.

    Independent of whether the event truth-matched, because it is the
    denominator for the mass response of *every* method, not only the matched
    ones. Takes the first parent in the record - the pair is degenerate.
    """
    pdg, mass = _gen_arrays(chunk, prefix, ("pdgId", "mass"))
    is_parent = np.isin(np.abs(pdg), np.asarray(spec.parent_pdgids))
    any_parent = is_parent.any(axis=1)
    first = np.argmax(is_parent, axis=1)
    rows = np.arange(pdg.shape[0])
    return np.where(any_parent, mass[rows, first], np.nan)


def _wrap_phi(dphi):
    """Fold a phi difference into [-pi, pi]."""
    wrapped = np.abs(dphi) % (2.0 * np.pi)
    return np.minimum(wrapped, 2.0 * np.pi - wrapped)
