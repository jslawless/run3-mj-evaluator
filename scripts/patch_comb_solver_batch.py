#!/usr/bin/env python3
"""patch_comb_solver_batch.py - make a batch-1-pinned CombSolver ONNX batchable.

Some CombinatorialSolver models (e.g. ml_model_final_*.onnx) were exported from a
batch=1 trace, which froze the trace's batch size into a couple of Reshape shape
constants and pinned the graph output to [1, 70]. The graph body is fully
batch-capable (the grouping scorer produces (batch, 7, 10)); only these traced
constants force a single event, so feeding >1 event trips an internal Reshape
('gemm_input_reshape': input {7, B, 256} cannot become {7, 256}).

This rewrites the two offending shape constants to use a dynamic batch axis and
relaxes the output's leading dim, producing an equivalent model that batches:

  * QKV-split reshape  [S, 1, 3, H]  ->  [S, -1, 3, H]   (MultiheadAttention)
  * final flatten      [1, -1]       ->  [-1, <n_logits>] (per-event logits)
  * graph output       [1, N]        ->  ['batch', N]

NOTE: the patched model must be loaded with the MatMulAddFusion optimizer
disabled (onnxruntime fuses MatMul+Add -> Gemm and re-inserts a batch-1
'gemm_input_reshape'). The evaluator does this automatically for comb_solver
models; if you load it yourself pass
    InferenceSession(..., disabled_optimizers=["MatMulAddFusion"]).

The real fix is to re-export the model with dynamic_axes; this is a transparent,
reproducible stop-gap when the training code is not at hand.

Usage:
    python patch_comb_solver_batch.py in.onnx out.onnx [--no-verify]
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def _reshape_const_targets(graph):
    """Map initializer-name -> list of Reshape nodes that use it as their shape."""
    by_init = {}
    for n in graph.node:
        if n.op_type == "Reshape" and len(n.input) >= 2:
            by_init.setdefault(n.input[1], []).append(n)
    return by_init


def patch(model):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    used_as_shape = _reshape_const_targets(g)
    n_logits = g.output[0].type.tensor_type.shape.dim[-1].dim_value or -1

    changed = []
    for name, init in inits.items():
        if name not in used_as_shape:
            continue
        val = numpy_helper.to_array(init)
        if val.ndim != 1:
            continue
        v = val.astype(np.int64).tolist()
        # MultiheadAttention QKV split: [seq, 1, 3, hidden] -> [seq, -1, 3, hidden]
        if len(v) == 4 and v[1] == 1 and v[2] == 3:
            new = [v[0], -1, v[2], v[3]]
        # Final per-event flatten before the logit gather: [1, -1] -> [-1, n_logits]
        elif v == [1, -1]:
            new = [-1, n_logits]
        else:
            continue
        init.CopyFrom(numpy_helper.from_array(np.array(new, dtype=np.int64), name))
        changed.append((name, v, new))

    # Relax the output's leading (batch) dim so it is no longer pinned to 1.
    od = g.output[0].type.tensor_type.shape.dim[0]
    out_was = od.dim_value if od.HasField("dim_value") else od.dim_param
    od.ClearField("dim_value")
    od.dim_param = "batch"

    return changed, out_was


def verify(orig_path, patched_model, n=16, seed=0):
    """Batched patched output must equal the original looped per-event output."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 7, 4)).astype(np.float32)
    o = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    truth = np.stack([o.run(None, {"four_momenta": x[i : i + 1]})[0][0] for i in range(n)])

    so = ort.SessionOptions()
    so.log_severity_level = 4
    p = ort.InferenceSession(
        patched_model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=["MatMulAddFusion"],
    )
    got = p.run(None, {"four_momenta": x})[0]
    same_argmax = bool((got.argmax(1) == truth.argmax(1)).all())
    max_abs = float(np.abs(got - truth).max())
    return same_argmax, max_abs, got.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the batched-vs-looped equivalence check.")
    args = ap.parse_args()

    model = onnx.load(args.input)
    changed, out_was = patch(model)
    if not changed:
        sys.exit("No batch-1 reshape constants found to patch; is this the right model?")
    print(f"Patched {args.input}:")
    for name, old, new in changed:
        print(f"  {name}: {old} -> {new}")
    print(f"  output[0] leading dim: {out_was} -> 'batch'")

    onnx.checker.check_model(model)

    if not args.no_verify:
        same, max_abs, shape = verify(args.input, model)
        print(f"Verify: batched output {shape}, argmax matches looped={same}, "
              f"max|Δlogits|={max_abs:.2e}")
        if not same:
            sys.exit("Verification FAILED: batched argmax differs from looped output.")

    onnx.save(model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
