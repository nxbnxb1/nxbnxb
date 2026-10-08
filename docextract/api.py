"""FastAPI backend.

POST /v1/extract            synchronous extraction (small documents)
POST /v1/jobs               asynchronous extraction; poll GET /v1/jobs/{id}
GET  /v1/jobs/{id}/markdown Markdown of a finished job
GET  /health                engines and settings in use
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings
from .pipeline import DocumentPipeline, ExtractOptions, parse_pages
from .preprocessing.loader import UnsupportedFormatError

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class _State:
    pipeline: DocumentPipeline | None = None
    lock = threading.Lock()
    jobs: dict[str, dict[str, Any]] = {}
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


def get_pipeline() -> DocumentPipeline:
    with _State.lock:
        if _State.pipeline is None:
            _State.pipeline = DocumentPipeline(Settings.from_env())
        return _State.pipeline


def set_pipeline(pipeline: DocumentPipeline) -> None:
    """Inject a pipeline (tests, custom engines)."""
    with _State.lock:
        _State.pipeline = pipeline


app = FastAPI(title="docextract", version=__version__, description="PDF / Word / scans → Markdown + JSON")


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    return data


def _options(pages: str | None, use_vlm: bool) -> ExtractOptions:
    try:
        return ExtractOptions(pages=parse_pages(pages), use_vlm=use_vlm)
    except ValueError as exc:
        raise HTTPException(422, f"invalid pages: {pages}") from exc


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "version": __version__, "engines": pipeline.engines.describe()}


@app.post("/v1/extract")
async def extract(
    file: UploadFile = File(...),
    pages: str | None = Form(None),
    use_vlm: bool = Form(True),
    include_json: bool = Form(True),
) -> dict:
    data = await _read(file)
    options = _options(pages, use_vlm)
    try:
        result = await run_in_threadpool(get_pipeline().process_bytes, data, file.filename or "upload", options)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    body: dict[str, Any] = {"markdown": result.markdown, "stats": result.stats.model_dump()}
    if include_json:
        body["document"] = result.model_dump(mode="json", exclude={"markdown"})
    return body


@app.post("/v1/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...), pages: str | None = Form(None), use_vlm: bool = Form(True)
) -> dict:
    data = await _read(file)
    options = _options(pages, use_vlm)
    job_id = uuid.uuid4().hex
    _State.jobs[job_id] = {"status": "queued", "filename": file.filename}

    def run() -> None:
        _State.jobs[job_id]["status"] = "running"
        try:
            result = get_pipeline().process_bytes(data, file.filename or "upload", options)
            _State.jobs[job_id].update(status="done", result=result)
        except Exception as exc:
            _State.jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")

    _State.pool.submit(run)
    return {"id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _State.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    body = {"id": job_id, "status": job["status"], "filename": job.get("filename")}
    if job["status"] == "done":
        result = job["result"]
        body["stats"] = result.stats.model_dump()
        body["document"] = result.model_dump(mode="json", exclude={"markdown"})
    if job["status"] == "failed":
        body["error"] = job["error"]
    return body


@app.get("/v1/jobs/{job_id}/markdown", response_class=PlainTextResponse)
def get_job_markdown(job_id: str) -> str:
    job = _State.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return job["result"].markdown or ""
