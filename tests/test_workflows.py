"""The training rounds of _train.yml are generated: the file must match the generator."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_rounds_are_up_to_date():
    check = subprocess.run([sys.executable, "scripts/gen_train_rounds.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout
