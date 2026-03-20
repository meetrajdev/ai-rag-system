from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import RagConfig
from rag.pipeline import run_rag
from rag.store import VectorStoreManager


class AskRequest(BaseModel):
    question: str


def register_routes(app: FastAPI, *, cfg: RagConfig, store: VectorStoreManager) -> None:
    """
    Attach API routes to the provided FastAPI app.
    """

    @app.post("/ask")
    async def ask(req: AskRequest):
        try:
            return run_rag(question=req.question, cfg=cfg, store=store)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

