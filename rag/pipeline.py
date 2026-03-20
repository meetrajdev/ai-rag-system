from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from .config import RagConfig
from .store import VectorStoreManager


def _make_snippet(text: str, max_chars: int = 240) -> str:
    # Collapse whitespace so snippets are readable in UI.
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def run_rag(*, question: str, cfg: RagConfig, store: VectorStoreManager) -> Dict[str, Any]:
    """
    End-to-end RAG:
    - reindex if docs changed
    - retrieve top-k chunks (with scores)
    - call LLM with retrieved context
    - return answer + citations (score + snippet)
    """
    store.maybe_reindex()

    # Returns List[(Document, score)]
    docs_with_scores = store.similarity_search_with_score(question)

    context_blocks: List[str] = []
    for i, (d, score) in enumerate(docs_with_scores, start=1):
        md = dict(getattr(d, "metadata", {}) or {})
        src = md.get("source") or "unknown_source"
        context_blocks.append(f"[{i}] score={score} source={src}\n{d.page_content}")
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Use the provided CONTEXT to answer.\n"
        "If the answer is not in the context, say you don't know based on the docs.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    llm = ChatOpenAI(model=cfg.openai_model, temperature=0)
    answer_msg = llm.invoke(prompt)
    answer = getattr(answer_msg, "content", str(answer_msg))

    citations: List[Dict[str, Any]] = []
    for i, (d, score) in enumerate(docs_with_scores, start=1):
        md = dict(getattr(d, "metadata", {}) or {})
        citations.append(
            {
                "rank": i,
                "score": float(score),
                "source": md.get("source"),
                "snippet": _make_snippet(getattr(d, "page_content", "")),
            }
        )

    return {"openai_response": answer, "citations": citations}

