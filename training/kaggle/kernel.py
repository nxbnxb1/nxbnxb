"""Kaggle kernel that trains the OCR model of one product on a free Kaggle GPU.

Pushed and collected by .github/workflows/_train_kaggle.yml (training/kaggle/make_kernel.py
fills in CONFIG). It checks out this repository at the pushed commit, installs PaddlePaddle for
the GPU (CPU with the fast CPU settings if no GPU works), runs training/vi_ocr/run_training.sh,
evaluates the model against the product's baseline and leaves in /kaggle/working:

    <model>.tar.gz       exported inference model (the Release asset Extract/Benchmark download)
    checkpoint.tar.gz    weights + dictionary + model.env (to continue training)
    report.md/.json      evaluation against the baseline
    train.log, run.json  training log and run summary
"""

import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

CONFIG = json.loads(r"""__CONFIG__""")

OUT = Path("/kaggle/working")
SRC = Path("/tmp/src")
WORK = Path("/tmp/work")
POCR = Path("/tmp/PaddleOCR")
FONTS = (
    "fonts-dejavu-core fonts-liberation2 fonts-freefont-ttf fonts-noto-core fonts-crosextra-carlito "
    "fonts-crosextra-caladea fonts-roboto fonts-open-sans fonts-noto-cjk fonts-ipaexfont-gothic fonts-ipaexfont-mincho"
)


def sh(cmd: str, check: bool = True, env: dict | None = None) -> int:
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, env=env).returncode


def main() -> None:
    start = time.time()
    run = {"config": CONFIG}
    sh(f"apt-get update -qq && apt-get install -y -qq {FONTS} >/dev/null", check=False)
    sh(f"git clone -q {CONFIG['repo']} {SRC} && git -C {SRC} checkout -q {CONFIG['sha']}")
    sh(
        f"git init -q {POCR} && git -C {POCR} fetch -q --depth 1 https://github.com/PaddlePaddle/PaddleOCR "
        f"{CONFIG['paddleocr_ref']} && git -C {POCR} checkout -q FETCH_HEAD"
    )

    gpu = shutil.which("nvidia-smi") is not None and sh("nvidia-smi", check=False) == 0
    if gpu:
        gpu = sh(f"pip install -q paddlepaddle-gpu=={CONFIG['paddle']} -i {CONFIG['paddle_gpu_index']}", check=False) == 0
        gpu = gpu and sh(
            "python -c \"import paddle; assert paddle.device.is_compiled_with_cuda(); paddle.utils.run_check()\"",
            check=False,
        ) == 0
    if not gpu:
        print("no usable GPU: training on CPU with the fast CPU settings", flush=True)
        sh(f"pip install -q paddlepaddle=={CONFIG['paddle']}")
    run["gpu"] = gpu
    sh(f"pip install -q paddleocr=={CONFIG['paddleocr']} wordfreq fonttools rapidfuzz")
    sh(f"pip install -q -r {POCR}/requirements.txt")
    # albumentations pulls opencv-python-headless, which shadows the cv2 build PaddleX expects
    sh("pip uninstall -y -q opencv-python-headless opencv-python", check=False)
    sh("pip install -q --force-reinstall --no-deps opencv-contrib-python==4.10.0.84")

    env = dict(
        os.environ,
        WORK=str(WORK),
        PADDLEOCR_DIR=str(POCR),
        LANGS=CONFIG["langs"],
        SAMPLES=str(CONFIG["samples"]),
        SEED="1",
        EPOCHS=str(CONFIG["epochs"]),
        CHUNK=str(CONFIG["chunk"]),
        TIME_BUDGET_MIN=str(CONFIG["minutes"]),
        VAL_COUNT="300",
        EVAL_COUNT="1000",
        PRINT_STEP="50",
        LOADER_WORKERS=str(os.cpu_count() or 4),
    )
    if gpu:  # the full model as PaddleOCR trains it
        env.update(USE_GPU="1", BATCH="128", FUSE="0", FREEZE="", GTC="1")
    else:  # settings measured by "OCR training · speed probe"
        env.update(USE_GPU="0", BATCH="32", FUSE="1", FREEZE="conv1,blocks2,blocks3,blocks4", GTC="0")

    if CONFIG.get("resume_url"):  # continue from the weights of a Release
        resume = WORK / "resume_dl"
        resume.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CONFIG["resume_url"], resume / "checkpoint.tar.gz")
        with tarfile.open(resume / "checkpoint.tar.gz") as tar:
            tar.extractall(resume, filter="data")
        ckpt = resume / "checkpoint"
        model_env = dict(line.split("=", 1) for line in (ckpt / "model.env").read_text().split() if "=" in line)
        weights = next(p for p in (ckpt / "weights.pdparams", ckpt / "latest.pdparams") if p.exists())
        env.update(INIT_PARAMS=str(weights), RESUME_DICT=str(ckpt / "dict.txt"), FUSE=model_env.get("FUSE", "0"))

    script = SRC / "training/vi_ocr/run_training.sh"
    sh(f"{script} prepare", env=env)
    sh(f"{script} train", env=env)
    best = WORK / "output/best_accuracy.pdparams"
    env["EXPORT_PARAMS"] = str(best if best.exists() else WORK / "output/final.pdparams")
    sh(f"{script} export", env=env)

    model_env = dict(line.split("=", 1) for line in (WORK / "model.env").read_text().split() if "=" in line)
    name, base = model_env["NAME"], model_env["BASE_MODEL"]
    sets = " ".join(f"{WORK}/data/eval_{lang}" for lang in CONFIG["langs"].split(","))
    sh(
        f"cd {SRC}/training/vi_ocr && python evaluate.py --eval {sets} {WORK}/data/eval_clean --baseline {base} "
        f"--model '{name} (fine-tuned)={WORK}/export' --model-name {base} --report {OUT}/report.md",
        check=False,
    )

    dist = WORK / "dist"
    (dist / "checkpoint").mkdir(parents=True, exist_ok=True)
    shutil.copy(env["EXPORT_PARAMS"], dist / "checkpoint/weights.pdparams")
    shutil.copy(WORK / "dict.txt", dist / "checkpoint/dict.txt")
    shutil.copy(WORK / "model.env", dist / "checkpoint/model.env")
    shutil.copytree(WORK / "export", dist / name)
    with tarfile.open(OUT / "checkpoint.tar.gz", "w:gz") as tar:
        tar.add(dist / "checkpoint", arcname="checkpoint")
    with tarfile.open(OUT / f"{name}.tar.gz", "w:gz") as tar:
        tar.add(dist / name, arcname=name)
    if (WORK / "output/train.log").exists():
        shutil.copy(WORK / "output/train.log", OUT / "train.log")
    run.update(model=name, baseline=base, minutes=round((time.time() - start) / 60, 1),
               weights=Path(env["EXPORT_PARAMS"]).name)
    (OUT / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1), flush=True)


if __name__ == "__main__":
    main()
