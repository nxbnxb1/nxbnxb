"""Adapt pretrained PP-OCRv5 recognition weights to the Vietnamese + English dictionary.

Only the output layers depend on the dictionary: the CTC classifier and the NRTR
(GTC) embedding/projection. Rows of characters both dictionaries share are copied;
each new Vietnamese letter starts from the closest letter the model already knows
(``ộ`` ← ``ô`` ← ``o``), so fine-tuning only has to learn the extra marks.
"""

from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import paddle

CTC_W = "head.ctc_head.fc.weight"  # [hidden, classes]
CTC_B = "head.ctc_head.fc.bias"  # [classes]
GTC_EMB = "head.gtc_head.embedding.embedding.weight"  # [classes, dim]
GTC_PRJ = "head.gtc_head.tgt_word_prj.weight"  # [dim, classes]


def read_dict(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.rstrip("\n\r")]


def fallbacks(ch: str) -> list[str]:
    """Closest known characters: drop the tone mark first, then every mark."""
    out = []
    decomposed = unicodedata.normalize("NFD", ch)
    tones = {"̀", "́", "̃", "̉", "̣"}
    without_tone = unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in tones))
    if without_tone != ch:
        out.append(without_tone)
    bare = "".join(c for c in decomposed if not unicodedata.combining(c))
    out.append({"đ": "d", "Đ": "D"}.get(bare, bare))
    return out


def remap(old: np.ndarray, old_chars: list[str], new_chars: list[str], axis: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    index = {c: i for i, c in enumerate(old_chars)}
    shape = list(old.shape)
    shape[axis] = len(new_chars)
    new = np.zeros(shape, dtype=old.dtype)
    scale = float(old.std()) * 0.02
    report = {"copied": 0, "from_base": 0, "random": 0}
    for j, ch in enumerate(new_chars):
        src = index.get(ch)
        kind = "copied"
        if src is None:
            kind = "from_base"
            src = next((index[b] for b in fallbacks(ch) if b in index), None)
        if src is None:
            kind = "random"
            row = rng.normal(0, float(old.std()), size=np.take(old, 0, axis=axis).shape)
        else:
            row = np.take(old, src, axis=axis)
            if kind == "from_base":
                row = row + rng.normal(0, scale, size=row.shape)
        if axis == 0:
            new[j] = row
        else:
            new[:, j] = row
        report[kind] += 1
    return new, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, help="e.g. latin_PP-OCRv5_mobile_rec_pretrained.pdparams")
    parser.add_argument("--old-dict", required=True, help="dictionary the pretrained model was trained with")
    parser.add_argument("--new-dict", required=True, help="vi_en_dict.txt")
    parser.add_argument("--out", required=True, help="output .pdparams")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    state = paddle.load(args.pretrained)
    old_dict, new_dict = read_dict(args.old_dict), read_dict(args.new_dict)

    # Class lists exactly as PaddleOCR builds them (use_space_char=True).
    old_ctc, new_ctc = ["blank"] + old_dict + [" "], ["blank"] + new_dict + [" "]
    specials = ["blank", "<unk>", "<s>", "</s>"]
    old_gtc, new_gtc = specials + old_dict + [" "], specials + new_dict + [" "]

    ctc_w = state[CTC_W].numpy()
    if ctc_w.shape[1] != len(old_ctc):
        raise SystemExit(f"CTC head has {ctc_w.shape[1]} classes, old dictionary gives {len(old_ctc)}")
    state[CTC_W], rep = remap(ctc_w, old_ctc, new_ctc, axis=1, rng=rng)
    state[CTC_B], _ = remap(state[CTC_B].numpy(), old_ctc, new_ctc, axis=0, rng=rng)
    print("CTC head:", rep)

    if GTC_EMB in state:
        emb = state[GTC_EMB].numpy()
        extra = emb.shape[0] - len(old_gtc)  # NRTR reserves trailing rows beyond the class list
        new_emb, rep = remap(emb[: len(old_gtc)], old_gtc, new_gtc, axis=0, rng=rng)
        state[GTC_EMB] = np.concatenate([new_emb, emb[len(old_gtc) :]], axis=0) if extra > 0 else new_emb
        prj = state[GTC_PRJ].numpy()
        new_prj, _ = remap(prj[:, : len(old_gtc)], old_gtc, new_gtc, axis=1, rng=rng)
        state[GTC_PRJ] = np.concatenate([new_prj, prj[:, len(old_gtc) :]], axis=1) if extra > 0 else new_prj
        print("NRTR head:", rep, f"(+{extra} reserved rows kept)")

    paddle.save({k: paddle.to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
