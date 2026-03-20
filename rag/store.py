import os
import threading
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .config import RagConfig
from .docs import chunk_documents, docs_signature, load_documents


class VectorStoreManager:
    """
    Owns the FAISS vectorstore and handles automatic reindexing when docs change.

    Reindexing is done on-demand (during a request) so we don't need a background file watcher.
    """

    def __init__(self, *, config: RagConfig):
        self._cfg = config
        self._lock = threading.Lock()

        # Embeddings model used by LangChain FAISS.
        self._embeddings = OpenAIEmbeddings(model=self._cfg.openai_embedding_model)

        # Track doc changes.
        self._last_sig = docs_signature(self._cfg.docs_dir)

        # Load or build store initially.
        self._vectorstore = self._load_or_build()

    def _load_or_build(self) -> FAISS:
        """
        Load an existing FAISS store from disk; if missing or incompatible, rebuild from docs.
        """
        if os.path.isdir(self._cfg.vector_store_dir):
            try:
                return FAISS.load_local(
                    self._cfg.vector_store_dir,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                # Fall through to rebuild
                pass

        os.makedirs(self._cfg.vector_store_dir, exist_ok=True)
        docs = load_documents(self._cfg.docs_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=self._cfg.chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
        )
        vs = FAISS.from_documents(chunks, self._embeddings)
        vs.save_local(self._cfg.vector_store_dir)
        return vs

    def maybe_reindex(self) -> None:
        """
        Rebuild the vector store if the docs folder fingerprint changed.
        """
        current = docs_signature(self._cfg.docs_dir)
        if current == self._last_sig:
            return

        with self._lock:
            # Double-check under lock to avoid duplicate rebuilds.
            current2 = docs_signature(self._cfg.docs_dir)
            if current2 == self._last_sig:
                return

            docs = load_documents(self._cfg.docs_dir)
            chunks = chunk_documents(
                docs,
                chunk_size=self._cfg.chunk_size,
                chunk_overlap=self._cfg.chunk_overlap,
            )
            vs = FAISS.from_documents(chunks, self._embeddings)
            os.makedirs(self._cfg.vector_store_dir, exist_ok=True)
            vs.save_local(self._cfg.vector_store_dir)

            self._vectorstore = vs
            self._last_sig = current2

    def similarity_search_with_score(self, query: str) -> List[Tuple[object, float]]:
        """
        Return (Document, score) pairs from FAISS.

        Note: the score meaning depends on the underlying FAISS distance strategy.
        Treat it as a relative relevance signal (rank + score), not a literal accuracy %.
        """
        return self._vectorstore.similarity_search_with_score(query, k=self._cfg.top_k)

