import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    """
    Centralized configuration for the RAG app.
    Values come from environment variables (optionally loaded from .env by `main.py`).
    """

    openai_model: str
    openai_embedding_model: str
    docs_dir: str
    vector_store_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_config(project_root: str) -> RagConfig:
    """
    Build config with sensible defaults.
    `project_root` should be the directory containing the project files.
    """
    return RagConfig(
        openai_model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        docs_dir=os.environ.get("DOCS_DIRECTORY") or os.path.join(project_root, "docs"),
        vector_store_dir=os.environ.get("VECTOR_STORE_DIR") or os.path.join(project_root, "vector_store"),
        chunk_size=int(os.environ.get("CHUNK_SIZE") or "1000"),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP") or "200"),
        top_k=int(os.environ.get("TOP_K") or "5"),
    )

