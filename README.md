## AI RAG System (FastAPI + LangChain + FAISS)

This project exposes a simple API that answers questions using **your local documents** via a **RAG** pipeline:

- Loads `.txt` and `.pdf` files from `./docs`
- Splits them into chunks
- Creates OpenAI embeddings (via LangChain `OpenAIEmbeddings`)
- Stores vectors locally in **FAISS** (`./vector_store`)
- Retrieves relevant chunks at query time and calls the LLM with that context

It also supports **automatic reindexing**: when you add/update/remove files in `docs/`, the index is rebuilt on the next `/ask` request.

---

## Project structure

- `main.py`: FastAPI app + `/ask` endpoint
- `app.py`: compatibility shim (so `uvicorn app:app` still works)
- `api/routes.py`: API routes (includes `/ask`)
- `rag/config.py`: configuration (env vars + defaults)
- `rag/docs.py`: doc loading, chunking, docs fingerprinting
- `rag/store.py`: FAISS load/save + auto reindex manager
- `rag/pipeline.py`: retrieval + prompt + answer + citations
- `docs/`: put your `.txt` / `.pdf` files here
- `vector_store/`: FAISS index saved locally

---

## Setup

1) Create and activate a virtual environment (recommended).

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DOCS_DIRECTORY=./docs
VECTOR_STORE_DIR=./vector_store
```

---

## Run the server

Run either of these (both work):

```bash
uvicorn main:app --reload
```

```bash
uvicorn app:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`.

---

## Use the API

### Ask a question

Endpoint: `POST /ask`

Request body:

```json
{ "question": "What stack do we use?" }
```

Example curl (Windows `cmd` escaping may differ):

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What stack do we use?\"}"
```

Response includes:

- `openai_response`: the answer grounded in retrieved chunks
- `citations`: ranked source files used for retrieval

---

## Automatic reindexing behavior

The app computes a fingerprint of `docs/` (file path + modified time + size).
If that fingerprint changes, it rebuilds FAISS and saves it to `vector_store/` **on the next** `/ask` request.

---

## Notes

- Put documents in `docs/` and call `/ask` to query them.
- If you delete/rename many docs and want a clean rebuild, you can delete `vector_store/` and restart the server.
