# run3-mj-evaluator

ML evaluator for the 6-jet multijet analysis. Reads a slimmed ROOT file
(produced by `run3-mj-slimmer/slim.py`), runs one or more ONNX neural-net
models to predict which 3+3 jets belong to each pair-produced particle, and
writes a new ROOT file with the original branches intact plus new
`MLCandidate` collections.

## Pipeline position

```
ScoutingNanoAOD ROOT  →  slim.py  →  slimmed ROOT  →  evaluate.py  →  evaluated ROOT
```

## Usage

```bash
python evaluate.py input.root output.root config.json
python evaluate.py input.root output.root config.json --tree events --chunk-size 50000
python evaluate.py mc.root output.root config.json --run-on-genjets
```

### Optional flags

| Flag | Description |
|---|---|
| `--tree NAME` | Input TTree name (default: `events`). |
| `--chunk-size N` | Events per processing chunk (default: `50000`). |
| `--run-on-genjets` | Also run every configured model on the `GenJet` collection (MC only). See [GenJet support](#genjet-support-mc) below. |

## Config format

```json
{
    "metadata": {"version": "v1"},
    "models": [
        {
            "type":         "spanet",
            "path":         "models/cart_model.onnx",
            "label":        "SPANet",
            "input_format": "cart"
        },
        {
            "type":       "comb_solver",
            "path":       "models/best_model.onnx",
            "label":      "CombSolver",
            "normalized": true
        }
    ]
}
```

### Model types

| `type` | Description | Required extra keys |
|---|---|---|
| `spanet` | SPANet ONNX model — batched inference, takes `(N, J, 4)` source + `(N, J)` mask | `input_format`: `"cart"` (px,py,pz,e) or `"spher"` (pt,eta,phi,e) |
| `comb_solver` | CombinatorialSolver ONNX — run one event at a time (batch=1 baked into graph), takes top-7 pT jets as `(1, 7, 4)` (E,px,py,pz) | `normalized`: `true` (divide by HT) or `false` (raw units) |

Gated PairNet exports from CombinatorialSolver (`pairnet_gated_7jet*.onnx`) also
use `"type": "comb_solver"`: same `(N, 7, 4)` input and 70-assignment order,
dynamic batch. A comb_solver's input width is read from its ONNX input
(e.g. `pairnet_gated_12jet_*` takes the leading 12 jets as context but, like
every comb_solver, pairs only the leading 7). Their pairings and `p_signal` are scale-invariant, so
`normalized` can be either. The optional `pairing_output` key picks the output
to argmax: `"logits"` (default, the gated drop-in) or `"signal_logits"` (signal
scorer alone, meant for use with a `p_signal` cut).

## Output ROOT file structure

### `events` TTree

All original branches from the slimmed file are copied through, plus for
each model in the config:

| Branch | Type | Description |
|---|---|---|
| `{label}Candidate_pt[2]` | float32 | pT of the two reconstructed trijet particles |
| `{label}Candidate_eta[2]` | float32 | η |
| `{label}Candidate_phi[2]` | float32 | φ |
| `{label}Candidate_mass[2]` | float32 | Invariant mass [GeV] |
| `{label}Candidate_jetIdx0[2]` | int32 | Index of 1st assigned jet in `ScoutingPFJet_*` arrays |
| `{label}Candidate_jetIdx1[2]` | int32 | Index of 2nd assigned jet |
| `{label}Candidate_jetIdx2[2]` | int32 | Index of 3rd assigned jet |
| `{label}Candidate_massAsymmetry` | float32 | \|m1 − m2\| / \|m1 + m2\| (one per event) |
| `{label}Candidate_pSignal` | float32 | Gated PairNet only: event-level P(signal) (one per event) |
| `{label}Candidate_mEst` | float32 | Mass-head PairNet only: learned resonance mass [GeV] (one per event) |

Each array has two elements per event: `[0]` = candidate 1, `[1]` = candidate 2.
The jet indices point into the per-event `ScoutingPFJet_*` arrays so you can
recover the constituent jet kinematics.

## GenJet support (MC)

If the input file contains a `GenJet` collection (nested `GenJet.pt` or flat
`GenJet_pt`/etc.), it is **always passed through** to the output unchanged —
no flag required. As with `ScoutingPFJet`, a flat layout is regrouped into a
single nested `GenJet` record on output (one shared offset counter).

Passing `--run-on-genjets` additionally runs every configured model a second
time using the `GenJet` 4-vectors as input. For each model this writes an
extra block of branches alongside the reco ones:

| Branch | Description |
|---|---|
| `{label}GenCandidate_pt[2]`, `_eta[2]`, `_phi[2]`, `_mass[2]` | Trijet candidates reconstructed from GenJets |
| `{label}GenCandidate_jetIdx0[2]`, `_jetIdx1[2]`, `_jetIdx2[2]` | Indices into the per-event `GenJet_*` arrays |

This is useful for truth-level closure tests and for comparing the model's
behaviour on parton-level vs. detector-level jets.

Notes:
- `--run-on-genjets` requires a `GenJet` collection in the input; the run
  aborts otherwise.
- Events with fewer than 7 GenJets are still processed (the input is padded
  with zero 4-vectors and the SPANet mask is set correctly), but indices and
  candidate kinematics for those events may not be physically meaningful —
  filter on GenJet multiplicity downstream if needed.

### `meta` TTree

One-row tree with evaluator version, config path, input file path, and
semicolon-joined model paths/labels.

### `cutflow` histogram

Passed through from the input file unchanged (when present).

## Dependencies

```
uproot >= 5
awkward >= 2
onnxruntime
numpy
```

The slimmed input file must contain at least `ScoutingPFJet_pt`,
`ScoutingPFJet_eta`, `ScoutingPFJet_phi`, and either
`ScoutingPFJet_px/py/pz/e` (preferred) or `ScoutingPFJet_m`.
`slim.py` computes and stores `px/py/pz/e` automatically.
