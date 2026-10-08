"""Average the weights of models with the same structure (one per runner, see _train.yml).

    python average_params.py OUT.pdparams IN1.pdparams IN2.pdparams ...

Runners start each round from the same weights and train on different synthetic lines; the
element-wise mean of their weights is the starting point of the next round (local SGD).
Integer tensors (counters) are taken from the first model.
"""

from __future__ import annotations

import sys

import numpy as np
import paddle


def main(out: str, inputs: list[str]) -> None:
    states = [paddle.load(path) for path in inputs]
    keys = list(states[0])
    for state, path in zip(states[1:], inputs[1:]):
        if list(state) != keys:
            raise SystemExit(f"{path} has other weights than {inputs[0]}")
    averaged = {}
    for key in keys:
        first = states[0][key]
        values = [np.asarray(s[key].numpy() if hasattr(s[key], "numpy") else s[key]) for s in states]
        if np.issubdtype(values[0].dtype, np.floating):
            mean = np.mean(np.stack([v.astype(np.float64) for v in values]), axis=0).astype(values[0].dtype)
            averaged[key] = paddle.to_tensor(mean)
        else:
            averaged[key] = first
    paddle.save(averaged, out)
    spread = max(
        float(np.max(np.abs(np.asarray(s[k].numpy()) - averaged[k].numpy())))
        for s in states
        for k in keys
        if np.issubdtype(np.asarray(s[k].numpy()).dtype, np.floating)
    )
    print(f"averaged {len(inputs)} models ({len(keys)} tensors) into {out}; largest deviation from the mean {spread:.4g}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
