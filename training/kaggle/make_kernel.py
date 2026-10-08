"""Write the Kaggle kernel (kernel-metadata.json + kernel.py) that trains one product's OCR model.

    python training/kaggle/make_kernel.py OUT_DIR --user KAGGLE_USER --product vi_en --repo URL --sha SHA \
        [--samples 200000 --epochs 10 --chunk 20000 --minutes 240 --resume-url URL]

Prints the kernel id (<user>/<slug>).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PADDLEOCR_REF = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"  # same training code as the GitHub workflows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--user", required=True)
    parser.add_argument("--product", choices=["vi_en", "vi_en_ja"], required=True)
    parser.add_argument("--repo", required=True, help="git URL of this repository (public)")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--resume-url", default="")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = parser.parse_args()

    config = {
        "repo": args.repo,
        "sha": args.sha,
        "product": args.product,
        "langs": "vi,en,ja" if args.product == "vi_en_ja" else "vi,en",
        "samples": args.samples,
        "epochs": args.epochs,
        "chunk": args.chunk,
        "minutes": args.minutes,
        "resume_url": args.resume_url,
        "paddle": "3.3.1",
        "paddle_gpu_index": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        "paddleocr": "3.7.0",
        "paddleocr_ref": PADDLEOCR_REF,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    code = (HERE / "kernel.py").read_text(encoding="utf-8").replace("__CONFIG__", json.dumps(config))
    (out / "kernel.py").write_text(code, encoding="utf-8")
    slug = "docextract-ocr-" + args.product.replace("_", "-")
    metadata = {
        "id": f"{args.user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": args.accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(metadata["id"])


if __name__ == "__main__":
    main()
