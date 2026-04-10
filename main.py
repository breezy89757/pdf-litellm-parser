import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import fitz  # pymupdf
import litellm
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(
    title="pdf-litellm-parser",
    description="Analyze multiple PDF files using LiteLLM vision model",
)

MODEL = "gpt-5.4-mini"
DPI = 100
IMAGE_DETAIL = "low"
MAT = fitz.Matrix(DPI / 72, DPI / 72)


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


@app.post("/analyze")
async def analyze(
    files: List[UploadFile] = File(..., description="One or more PDF files"),
    prompt: str = Form(..., description="Prompt to send along with the PDF pages"),
):
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

    # Build message content array
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

    # Call LiteLLM
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LiteLLM error: {exc}") from exc

    return {
        "result": response.choices[0].message.content,
        "pages_processed": len(all_pages_b64),
        "model": MODEL,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
