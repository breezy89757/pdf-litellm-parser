# pdf-litellm-parser

A FastAPI service that analyzes multiple PDF files using LiteLLM vision model (gpt-5.4-mini).

## 專案說明

本服務提供單一 endpoint，接收多份 PDF 檔案及一個 prompt，將每頁 PDF 轉成 PNG 圖片後透過 LiteLLM 傳給視覺模型進行分析。

### 技術棧

- **Backend**: FastAPI
- **LLM**: gpt-5.4-mini via LiteLLM (vision capable)
- **PDF 處理**: PyMuPDF (fitz)
- **並行處理**: ThreadPoolExecutor

### 功能特色

- 接收多份 PDF（`multipart/form-data`，欄位名 `files`）
- 接收 prompt 字串（欄位名 `prompt`）
- 每份 PDF 使用 pymupdf (fitz) 以 DPI 100 將每頁轉成 PNG，base64 編碼
- Image detail 設為 `low`（~85 tokens/頁，適合文字為主 PDF，速度快）
- 多份 PDF 的頁面轉換使用 `ThreadPoolExecutor` 並行處理
- 所有頁面圖片組成 content array 連同 prompt 送給 LiteLLM

## 安裝方式

### 前置需求

- Python 3.9+
- OpenAI API Key（或其他 LiteLLM 支援的 provider）

### 步驟

```bash
# 進入專案目錄
cd projects/pdf-litellm-parser

# （建議）建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝相依套件
pip install -r requirements.txt
```

### 環境變數

```bash
export OPENAI_API_KEY=sk-...   # 或其他 LiteLLM 支援的 provider key
```

## 使用方式

```bash
# 啟動開發伺服器（預設 port 8000）
uvicorn main:app --reload
```

服務啟動後，可透過以下網址存取：

| 端點 | URL |
|------|-----|
| 分析 PDF | http://localhost:8000/analyze |
| 健康檢查 | http://localhost:8000/health |
| API 文件 | http://localhost:8000/docs |

### 範例請求

```bash
curl -X POST http://localhost:8000/analyze \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "prompt=請摘要這些文件的主要內容"
```

### 範例回應

```json
{
  "result": "這兩份文件主要討論...",
  "pages_processed": 5,
  "model": "gpt-5.4-mini"
}
```

## API 規格

### POST /analyze

| 欄位 | 類型 | 說明 |
|------|------|------|
| `files` | `File[]` | 一或多份 PDF 檔案（multipart/form-data） |
| `prompt` | `string` | 傳送給模型的提示詞 |

**回應**：

```json
{
  "result": "<模型回應>",
  "pages_processed": <處理頁數>,
  "model": "gpt-5.4-mini"
}
```

### GET /health

健康檢查端點，回傳 `{"status": "ok"}`。
