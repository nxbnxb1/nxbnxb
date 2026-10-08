"""VLM client for any OpenAI-compatible chat endpoint (vLLM / SGLang / LMDeploy serving Qwen-VL, ...)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from PIL import Image

from ..config import Settings
from .base import VlmRequest, VlmResult

log = logging.getLogger(__name__)

_RETRYABLE = frozenset({429, 500, 502, 503, 504})

_EXACT = (
    "Copy text exactly as printed: keep the original language and every diacritic, number, unit, "
    "symbol and punctuation mark. Never translate, summarise, correct or invent content. "
    "If something is unreadable write [illegible]."
)

# The VLM never transcribes running text (that is OCR's job); it only handles
# structure OCR cannot recover and visual content.
PROMPTS: dict[str, str] = {
    "table": (
        "Convert the table in this image to a single HTML <table>. Use <tr>, <th> for header cells and <td>, "
        "and rowspan/colspan for merged cells. " + _EXACT + " Keep empty cells empty. "
        "Output only the HTML table."
    ),
    "formula": (
        "Convert the mathematical formula in this image to LaTeX. Output only the LaTeX code, "
        "without $ delimiters or explanations."
    ),
    "chart": (
        "Analyse this chart for a document search index. Respond with JSON only, using the keys: "
        '"chart_type", "title", "description" (2-4 factual sentences: what is measured, trends, extremes), '
        '"columns" (list of column names) and "rows" (list of rows with the data values). '
        "Only report values printed on the chart or clearly readable from the axes; prefix estimated values "
        "with ~. Copy labels exactly. {language}"
    ),
    "image": (
        "Describe this image from a document in 1-3 factual sentences for a search index, then transcribe "
        "any text visible in it exactly. Do not speculate beyond what is visible. {language}"
    ),
}


_LANGUAGES = {"vi": "Vietnamese", "en": "English"}


def prompt_for(task: str, hint: str | None, language: str | None) -> str:
    lang = (
        f"Write the description in {_LANGUAGES.get(language, language)}."
        if language
        else "Write the description in Vietnamese if the image contains Vietnamese text, otherwise in English."
    )
    prompt = PROMPTS[task].replace("{language}", lang)
    if hint:
        prompt += (
            "\n\nText extracted from the same area of the original file (reliable characters, but layout may be "
            f"lost; use it to get exact spellings and numbers):\n<<<\n{hint[:4000]}\n>>>"
        )
    return prompt


def encode_image(image: Image.Image, max_side: int) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clean_output(text: str) -> str:
    """Strip code fences and chatty wrappers some models add."""
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text


def parse_json(text: str) -> dict | None:
    text = clean_output(text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class OpenAICompatibleVLM:
    """Thread-pooled client; requests in one ``run`` call are sent concurrently."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.vlm_base_url:
            raise ValueError("vlm_base_url is not set")
        self.settings = settings
        self.name = f"vlm:{settings.vlm_model}"
        headers = {"Content-Type": "application/json"}
        if settings.vlm_api_key:
            headers["Authorization"] = f"Bearer {settings.vlm_api_key}"
        self._client = client or httpx.Client(
            base_url=settings.vlm_base_url.rstrip("/"), headers=headers, timeout=settings.vlm_timeout
        )
        self._pool = ThreadPoolExecutor(max_workers=settings.vlm_max_concurrency, thread_name_prefix="vlm")

    def run(self, requests: list[VlmRequest]) -> list[VlmResult]:
        return list(self._pool.map(self._call, requests))

    def _call(self, request: VlmRequest) -> VlmResult:
        s = self.settings
        payload = {
            "model": s.vlm_model,
            "temperature": s.vlm_temperature,
            "max_tokens": s.vlm_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image(request.image, s.vlm_max_image_side)}},
                        {"type": "text", "text": prompt_for(request.task, request.hint, s.vlm_language)},
                    ],
                }
            ],
        }
        if request.task == "chart":
            payload["response_format"] = {"type": "json_object"}
        response = self._post(payload)
        body = response.json()
        text = clean_output(body["choices"][0]["message"].get("content") or "")
        usage = body.get("usage") or {}
        result = VlmResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
        if request.task == "chart":
            result.data = parse_json(text)
        return result

    def _post(self, payload: dict) -> httpx.Response:
        retries = 0
        while True:
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                error: Exception = exc
            else:
                if response.status_code == 400 and "response_format" in payload:
                    # Some servers reject response_format; the prompt already asks for JSON.
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    continue
                if response.status_code not in _RETRYABLE:
                    response.raise_for_status()
                    return response
                error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
            if retries >= self.settings.vlm_max_retries:
                raise error
            retries += 1
            delay = min(30.0, 2.0**retries)
            log.warning("VLM request failed (%s), retry %d in %.0fs", error, retries, delay)
            time.sleep(delay)
