"""PaddleOCR ``tools/train.py`` with options that matter on CPU-only runners.

Run from the root of a PaddleOCR checkout, with the same arguments as ``tools/train.py``.
Thread count and oneDNN are set through Paddle's environment flags by run_training.sh.

Env:
  FREEZE  comma-separated backbone stages kept fixed (``conv1,blocks2,blocks3,blocks4``).
          Their weights get no gradient, so their backward pass is skipped entirely; the
          early stages detect strokes and edges and transfer to new letters unchanged.
  GTC     ``0`` trains the CTC branch only (the NRTR guidance branch is not run). Inference
          uses the CTC branch in both cases, so the exported model has the same structure.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

import paddle  # noqa: E402

import tools.program as program  # noqa: E402
import tools.train as train  # noqa: E402
from ppocr.losses.rec_multi_loss import MultiLoss  # noqa: E402
from ppocr.modeling.heads.rec_multi_head import MultiHead  # noqa: E402
from ppocr.utils.utility import set_seed  # noqa: E402

FREEZE = [s.strip() for s in os.environ.get("FREEZE", "").split(",") if s.strip()]
GTC = os.environ.get("GTC", "1") != "0"


def _build_model(config, _build=train.build_model):
    model = _build(config)
    for name in FREEZE:
        for param in getattr(model.backbone, name).parameters():
            param.stop_gradient = True
    trainable = sum(int(p.numel()) for p in model.parameters() if not p.stop_gradient)
    total = sum(int(p.numel()) for p in model.parameters())
    print(f"train_cpu: frozen={FREEZE or 'none'} gtc={GTC} trainable={trainable}/{total} parameters")
    return model


def _ctc_only_forward(self, x, targets=None, _forward=MultiHead.forward):
    if not self.training:
        return _forward(self, x, targets)
    if self.use_pool:
        x = self.pool(x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2]))
    ctc_encoder = self.ctc_encoder(x)
    return {"ctc": self.ctc_head(ctc_encoder, targets), "ctc_neck": ctc_encoder}


def _ctc_only_loss(self, predicts, batch, _forward=MultiLoss.forward):
    if "gtc" in predicts or "sar" in predicts:
        return _forward(self, predicts, batch)
    loss = self.loss_funcs["CTCLoss"](predicts["ctc"], batch[:2] + batch[3:])["loss"] * self.weight_1
    self.total_loss = {"CTCLoss": loss, "loss": loss}
    return self.total_loss


def _flags() -> str:
    found = {}
    for flag in ("FLAGS_paddle_num_threads", "FLAGS_use_mkldnn", "FLAGS_use_onednn"):
        try:
            found.update(paddle.get_flags([flag]))
        except Exception:  # flag unknown to this Paddle version
            pass
    return ", ".join(f"{k}={v}" for k, v in found.items())


if __name__ == "__main__":
    train.build_model = _build_model
    if not GTC:
        MultiHead.forward = _ctc_only_forward
        MultiLoss.forward = _ctc_only_loss
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    logger.info(f"train_cpu: {_flags()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    set_seed(config["Global"].get("seed", 1024))
    train.main(config, device, logger, vdl_writer)
