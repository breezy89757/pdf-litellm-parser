import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

import fitz  # pymupdf
import litellm
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="pdf-litellm-parser",
    description="Analyze multiple PDF files using LiteLLM vision model",
)

MODEL = os.getenv("LITELLM_MODEL", "gpt-5.4-mini")
API_BASE = os.getenv("AOAI_API_BASE") or os.getenv("OPENAI_API_BASE")
API_KEY = os.getenv("AOAI_API_KEY") or os.getenv("OPENAI_API_KEY")
MAX_IMAGES_PER_REQUEST = int(os.getenv("LITELLM_MAX_IMAGES", "50"))
MAP_BATCH_SIZE = int(os.getenv("LITELLM_MAP_BATCH_SIZE", "20"))
MAP_BATCH_OVERLAP = int(os.getenv("LITELLM_MAP_BATCH_OVERLAP", "0"))
MAP_MAX_WORKERS = int(os.getenv("LITELLM_MAP_MAX_WORKERS", "2"))
MERGE_MAX_CHARS_PER_BATCH = int(os.getenv("LITELLM_MERGE_MAX_CHARS_PER_BATCH", "3000"))
LITELLM_TIMEOUT = float(os.getenv("LITELLM_TIMEOUT", "90"))
BATCH_MAX_TOKENS = int(os.getenv("LITELLM_BATCH_MAX_TOKENS", "900"))
MERGE_MAX_TOKENS = int(os.getenv("LITELLM_MERGE_MAX_TOKENS", "1200"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "LITELLM_SYSTEM_PROMPT",
    (
        "你是專業文件分析助理，嚴格根據提供頁面內容回答，不要推測。\n"
        "回答時必須：\n"
        "1) 優先保留具體數字、百分比、模型名稱\n"
        "2) 優先指出核心技術主張與可驗證事實\n"
        "3) 若資訊不足要明確標記不確定\n"
        "4) 盡量精簡，避免冗長敘述"
    ),
)
DPI = 100
IMAGE_DETAIL = "high"
MAT = fitz.Matrix(DPI / 72, DPI / 72)

if API_BASE and API_BASE.rstrip("/").endswith("/responses"):
    API_BASE = API_BASE.rstrip("/")[: -len("/responses")]


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version="0.1.0",
        description=app.description,
        routes=app.routes,
    )

    # Swagger UI currently renders OpenAPI 3.1 file arrays as text inputs.
    # Convert UploadFile array schema into OpenAPI 3.0-compatible binary format.
    body_schema = (
        schema.get("components", {})
        .get("schemas", {})
        .get("Body_analyze_analyze_post", {})
        .get("properties", {})
        .get("files", {})
    )
    items = body_schema.get("items")
    if isinstance(items, dict) and items.get("type") == "string":
        if items.get("contentMediaType") == "application/octet-stream":
            items.pop("contentMediaType", None)
            items["format"] = "binary"

    schema["openapi"] = "3.0.3"
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


def pdf_bytes_to_b64_pages(pdf_bytes: bytes) -> List[str]:
    """Convert all pages of a PDF (raw bytes) to base64-encoded PNG strings."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_b64: List[str] = []
    for page in doc:
        pix = page.get_pixmap(matrix=MAT)
        png_bytes = pix.tobytes("png")
        pages_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
    doc.close()
    return pages_b64


def batch_ranges(total_pages: int, batch_size: int, overlap: int) -> List[tuple[int, int]]:
    if total_pages <= 0:
        return []

    batch_size = max(1, batch_size)
    overlap = max(0, min(overlap, batch_size - 1))
    step = batch_size - overlap

    ranges: List[tuple[int, int]] = []
    start = 0
    while start < total_pages:
        end = min(start + batch_size, total_pages)
        ranges.append((start, end))
        if end >= total_pages:
            break
        start += step
    return ranges


def call_litellm(messages: list[dict], system_prompt: str | None = None, max_tokens: int | None = None):
    final_messages: list[dict[str, Any]] = []
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})
    final_messages.extend(messages)

    completion_kwargs = {
        "model": MODEL,
        "messages": final_messages,
        "timeout": LITELLM_TIMEOUT,
    }
    if max_tokens and max_tokens > 0:
        completion_kwargs["max_tokens"] = max_tokens
    if API_BASE:
        completion_kwargs["api_base"] = API_BASE
    if API_KEY:
        completion_kwargs["api_key"] = API_KEY
    return litellm.completion(**completion_kwargs)


def extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": a["prompt_tokens"] + b["prompt_tokens"],
        "completion_tokens": a["completion_tokens"] + b["completion_tokens"],
        "total_tokens": a["total_tokens"] + b["total_tokens"],
    }


def analyze_batch(
    batch_pages: List[str],
    user_prompt: str,
    focus_prompt: str,
    start_idx: int,
    end_idx: int,
    system_prompt: str,
) -> tuple[str, dict[str, int]]:
    focus_block = f"{focus_prompt}\n" if focus_prompt else ""
    content = [
        {
            "type": "text",
            "text": (
                "請嚴格根據這一批 PDF 頁面做資訊擷取，不要猜測。\\n"
                f"本批頁碼範圍（整份文件）: {start_idx + 1}-{end_idx}\\n"
                f"分析焦點: {focus_block}"
                f"使用者需求: {user_prompt}\\n"
                "請輸出（精簡條列）:\\n"
                "1) 核心主張（最多 4 點）\\n"
                "2) 關鍵數字/百分比/排名（最多 8 點，保留原始數值）\\n"
                "3) 重要模型/方法名稱\\n"
                "4) 不確定處"
            ),
        }
    ]
    for b64 in batch_pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": IMAGE_DETAIL,
                },
            }
        )

    response = call_litellm(
        messages=[{"role": "user", "content": content}],
        system_prompt=system_prompt,
        max_tokens=BATCH_MAX_TOKENS,
    )
    return response.choices[0].message.content, extract_usage(response)


def merge_batch_results(
    batch_results: List[str],
    user_prompt: str,
    focus_prompt: str,
    system_prompt: str,
) -> tuple[str, dict[str, int]]:
    merged_input = "\n\n".join(
        [f"[批次{i + 1}]\n{result[:MERGE_MAX_CHARS_PER_BATCH]}" for i, result in enumerate(batch_results)]
    )
    focus_block = f"{focus_prompt}\n" if focus_prompt else ""
    merge_prompt = (
        "你會看到多個分批擷取結果，請做整體整合。\n"
        "要求：\n"
        "1) 去除重複\n"
        "2) 保留彼此矛盾或不確定的點\n"
        "3) 輸出最終精簡摘要（先結論再重點）\n"
        "4) 必須保留跨批最關鍵的具體數字與核心 claim\n"
        f"分析焦點: {focus_block}"
        f"使用者需求: {user_prompt}\n\n"
        "以下是分批結果:\n"
        f"{merged_input}"
    )
    response = call_litellm(
        messages=[{"role": "user", "content": merge_prompt}],
        system_prompt=system_prompt,
        max_tokens=MERGE_MAX_TOKENS,
    )
    return response.choices[0].message.content, extract_usage(response)


@app.post("/analyze")
async def analyze(
    files: List[UploadFile] = File(..., description="One or more PDF files"),
    prompt: str = Form(..., description="Prompt to send along with the PDF pages"),
    focus_prompt: str = Form(
        default="",
        description="Optional task focus layered before user prompt (e.g. 核心技術貢獻、benchmark 數字)",
    ),
    system_prompt: str = Form(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt used to control analysis behavior",
    ),
):
    started_at = time.perf_counter()

    if not files:
        raise HTTPException(status_code=400, detail="No PDF files provided.")

    # Read all file bytes upfront (async) before handing off to threads
    file_bytes_list: List[bytes] = []
    for f in files:
        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty.")
        file_bytes_list.append(content)

    # Convert PDFs to page images in parallel
    all_pages_b64: List[str] = []
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(pdf_bytes_to_b64_pages, data): idx
            for idx, data in enumerate(file_bytes_list)
        }
        # Collect results preserving PDF order
        results: dict[int, List[str]] = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process PDF #{idx + 1}: {exc}",
                ) from exc

    for idx in sorted(results.keys()):
        all_pages_b64.extend(results[idx])

    if not all_pages_b64:
        raise HTTPException(status_code=422, detail="No pages could be extracted from the provided PDFs.")

    total_pages_extracted = len(all_pages_b64)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Single-call path for small documents; batch + merge path for larger documents.
    try:
        if total_pages_extracted <= MAX_IMAGES_PER_REQUEST:
            content = [{"type": "text", "text": prompt}]
            for b64 in all_pages_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": IMAGE_DETAIL,
                        },
                    }
                )
            response = call_litellm(
                messages=[{"role": "user", "content": content}],
                system_prompt=system_prompt,
                max_tokens=MERGE_MAX_TOKENS,
            )
            final_result = response.choices[0].message.content
            total_usage = extract_usage(response)
            batch_count = 1
            pages_processed = total_pages_extracted
            pages_truncated = 0
        else:
            effective_batch_size = min(MAX_IMAGES_PER_REQUEST, max(1, MAP_BATCH_SIZE))
            ranges = batch_ranges(total_pages_extracted, effective_batch_size, MAP_BATCH_OVERLAP)
            batch_outputs: dict[int, str] = {}
            batch_usage: dict[int, dict[str, int]] = {}

            with ThreadPoolExecutor(max_workers=max(1, MAP_MAX_WORKERS)) as executor:
                future_to_batch_idx = {}
                for batch_idx, (start, end) in enumerate(ranges):
                    future = executor.submit(
                        analyze_batch,
                        all_pages_b64[start:end],
                        prompt,
                        focus_prompt,
                        start,
                        end,
                        system_prompt,
                    )
                    future_to_batch_idx[future] = batch_idx

                for future in as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    try:
                        batch_result, usage = future.result()
                        batch_outputs[batch_idx] = batch_result
                        batch_usage[batch_idx] = usage
                    except Exception as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"LiteLLM batch #{batch_idx + 1} error: {exc}",
                        ) from exc

            ordered_outputs = [batch_outputs[i] for i in range(len(ranges))]
            for idx in range(len(ranges)):
                total_usage = add_usage(total_usage, batch_usage[idx])

            merged_result, merge_usage = merge_batch_results(
                ordered_outputs,
                prompt,
                focus_prompt,
                system_prompt,
            )
            final_result = merged_result
            total_usage = add_usage(total_usage, merge_usage)
            batch_count = len(ranges)
            pages_processed = total_pages_extracted
            pages_truncated = 0
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LiteLLM error: {exc}") from exc

    return {
        "result": final_result,
        "pages_processed": pages_processed,
        "pages_extracted_total": total_pages_extracted,
        "pages_truncated": pages_truncated,
        "batch_count": batch_count,
        "token_usage": total_usage,
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "image_detail": IMAGE_DETAIL,
        "model": MODEL,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
