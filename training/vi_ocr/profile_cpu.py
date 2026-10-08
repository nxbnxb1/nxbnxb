"""Where does a CPU training step of the recogniser spend its time?

Run from the root of a PaddleOCR checkout:

    python profile_cpu.py -c <config.yml> --classes 300 [--batch 32] [--torch]

Prints, for one training step on a random batch: forward time of every top-level part
(backbone stages, CTC neck/head, NRTR head), loss, backward and optimizer time; then the same
step with the backbone re-parameterised (each block's training branches fused into the single
convolution used at inference); then a micro-benchmark of the backbone's convolution shapes in
Paddle (with and without oneDNN) and, with --torch, in PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import paddle  # noqa: E402
import yaml  # noqa: E402

from ppocr.losses import build_loss  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402


def make_model(config: dict, classes: int, fused: bool):
    arch = copy.deepcopy(config["Architecture"])  # build_model consumes parts of the config
    arch["Head"]["out_channels_list"] = {"CTCLabelDecode": classes, "NRTRLabelDecode": classes + 3}
    model = build_model(arch)
    if fused:
        for layer in model.backbone.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
                layer.rep()
    model.train()
    return model


def make_batch(batch: int, classes: int):
    rng = np.random.default_rng(0)
    image = paddle.to_tensor(rng.standard_normal((batch, 3, 48, 320)).astype("float32"))
    label_ctc = paddle.to_tensor(rng.integers(1, classes, (batch, 25)).astype("int64"))
    label_gtc = paddle.to_tensor(rng.integers(3, classes, (batch, 27)).astype("int64"))
    length = paddle.to_tensor(np.full((batch,), 18, dtype="int64"))
    valid_ratio = paddle.to_tensor(np.ones((batch,), dtype="float32"))
    return [image, label_ctc, label_gtc, length, valid_ratio]


class Timer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self._start: dict[str, float] = {}

    def hook(self, model) -> None:
        parts = [(f"backbone.{n}", m) for n, m in model.backbone.named_children()]
        head = model.head
        for name in ("ctc_encoder", "ctc_head", "before_gtc", "gtc_head"):
            if hasattr(head, name):
                parts.append((f"head.{name}", getattr(head, name)))
        for name, layer in parts:
            layer.register_forward_pre_hook(lambda _l, _i, n=name: self._start.__setitem__(n, time.perf_counter()))
            layer.register_forward_post_hook(lambda _l, _i, _o, n=name: self._add(n))

    def _add(self, name: str) -> None:
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - self._start[name]


def profile_step(config: dict, classes: int, batch_size: int, fused: bool, steps: int) -> None:
    model = make_model(config, classes, fused)
    loss_fn = build_loss(copy.deepcopy(config["Loss"]))
    opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    batch = make_batch(batch_size, classes)
    timer = Timer()
    timer.hook(model)
    totals = {"forward": 0.0, "loss": 0.0, "backward": 0.0, "optimizer": 0.0}
    for step in range(steps + 1):
        if step == 1:  # first step = warm-up
            timer.times.clear()
            totals = dict.fromkeys(totals, 0.0)
        t0 = time.perf_counter()
        preds = model(batch[0], data=batch[1:])
        t1 = time.perf_counter()
        loss = loss_fn(preds, batch)["loss"]
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        opt.step()
        opt.clear_grad()
        t4 = time.perf_counter()
        for key, dt in zip(totals, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            totals[key] += dt
    label = "fused backbone" if fused else "as trained by PaddleOCR (training branches)"
    step_s = sum(totals.values()) / steps
    print(f"\n### one step, batch {batch_size}, {label}: {step_s:.2f} s = {batch_size / step_s:.1f} samples/s")
    print("| part | seconds/step |\n|---|---|")
    for key, value in totals.items():
        print(f"| **{key}** | {value / steps:.3f} |")
    for key, value in timer.times.items():
        print(f"| forward {key} | {value / steps:.3f} |")


def bench(fn, repeat: int = 5) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


# (batch, channels, height, width, kernel, groups): depthwise and pointwise convolutions of the backbone
SHAPES = [
    (32, 64, 12, 160, 3, 64),
    (32, 128, 6, 160, 5, 128),
    (32, 256, 3, 80, 5, 256),
    (32, 128, 6, 160, 1, 1),
]


def conv_bench(use_torch: bool) -> None:
    print("\n### convolution forward+backward, seconds")
    header = "| shape (N,C,H,W,k,groups) | paddle | paddle oneDNN |" + (" torch |" if use_torch else "")
    print(header + "\n|---|---|---|" + ("---|" if use_torch else ""))
    if use_torch:
        import torch

        torch.set_num_threads(os.cpu_count() or 1)
    for n, c, h, w, k, g in SHAPES:
        row = []
        for onednn in (False, True):
            try:
                paddle.set_flags({"FLAGS_use_mkldnn": onednn})
            except Exception:
                pass
            conv = paddle.nn.Conv2D(c, c, k, padding=k // 2, groups=g)
            x = paddle.randn([n, c, h, w])
            x.stop_gradient = False

            def run(conv=conv, x=x):
                conv(x).mean().backward()

            row.append(bench(run))
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass
        if use_torch:
            tconv = torch.nn.Conv2d(c, c, k, padding=k // 2, groups=g)
            tx = torch.randn(n, c, h, w, requires_grad=True)

            def trun(tconv=tconv, tx=tx):
                tconv(tx).mean().backward()

            row.append(bench(trun))
        print(f"| {(n, c, h, w, k, g)} | " + " | ".join(f"{v:.4f}" for v in row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    paddle.set_device("cpu")
    print(f"cores={os.cpu_count()} flags={paddle.get_flags(['FLAGS_paddle_num_threads'])}")
    config = yaml.safe_load(open(args.config))
    conv_bench(args.torch)
    profile_step(config, args.classes, args.batch, fused=True, steps=args.steps)
    profile_step(config, args.classes, args.batch, fused=False, steps=args.steps)


if __name__ == "__main__":
    main()
