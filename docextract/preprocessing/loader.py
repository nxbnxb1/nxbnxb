"""Input format detection."""

from __future__ import annotations

import io
import zipfile
from typing import Literal

from PIL import Image

from .office import office_target

Format = Literal["pdf", "docx", "image", "office"]


class UnsupportedFormatError(ValueError):
    pass


def detect_format(data: bytes, filename: str = "") -> Format:
    name = filename.lower()
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    if office_target(name):
        return "office"  # converted by LibreOffice (preprocessing/office.py)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return "image"
    except Exception:
        pass
    raise UnsupportedFormatError(
        f"{filename or 'file'}: unsupported format (PDF, Word, office documents, HTML, text or images)"
    )
