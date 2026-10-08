"""Every other document format goes through LibreOffice into the common pipeline.

Legacy Word (.doc, .rtf, .odt) becomes .docx (its structure is read natively); presentations,
spreadsheets, HTML and plain text become PDF with a text layer. One converter instead of a
reader per format.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# extension → target format
TO_DOCX = {".doc", ".rtf", ".odt", ".wpd"}
TO_PDF = {".ppt", ".pptx", ".odp", ".xls", ".xlsx", ".ods", ".csv", ".html", ".htm", ".txt", ".md", ".epub"}
OFFICE_EXTENSIONS = TO_DOCX | TO_PDF


def office_target(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in TO_DOCX:
        return "docx"
    if ext in TO_PDF:
        return "pdf"
    return None


def convert(data: bytes, filename: str, target: str, timeout: int = 600) -> bytes:
    """Convert with LibreOffice (``soffice --headless --convert-to``)."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ValueError(f"{filename}: converting this format needs LibreOffice (soffice)")
    ext = Path(filename).suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext}"
        src.write_bytes(data)
        try:
            subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", target, "--outdir", tmp, str(src)],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"{filename}: LibreOffice could not convert it ({exc})") from exc
        out = Path(tmp) / f"input.{target}"
        if not out.exists():
            raise ValueError(f"{filename}: LibreOffice produced no {target}")
        return out.read_bytes()
