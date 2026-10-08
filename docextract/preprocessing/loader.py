"""Input format detection."""

from __future__ import annotations

import io
import zipfile
from typing import Literal

from PIL import Image

Format = Literal["pdf", "docx", "doc", "image"]

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


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
        raise UnsupportedFormatError(f"{filename or 'file'}: ZIP container that is not a Word document")
    if data[:8] == _OLE_MAGIC and name.endswith(".doc"):
        return "doc"
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return "image"
    except Exception:
        pass
    raise UnsupportedFormatError(f"{filename or 'file'}: unsupported format (expected PDF, DOCX or an image)")
