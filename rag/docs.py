import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir: str):
    """
    Load all supported documents under `docs_dir` into LangChain Document objects.
    Supported: .txt, .pdf (recursively).
    """
    if not os.path.isdir(docs_dir):
        return []

    txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return txt_loader.load() + pdf_loader.load()


def chunk_documents(documents, *, chunk_size: int, chunk_overlap: int):
    """
    Split documents into overlapping chunks for retrieval/embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def iter_doc_files(docs_dir: str) -> List[str]:
    """
    Return a stable list of .txt/.pdf file paths under docs_dir.
    Used to detect when docs changed (for automatic reindexing).
    """
    if not os.path.isdir(docs_dir):
        return []

    paths: List[str] = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".txt") or lower.endswith(".pdf"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def docs_signature(docs_dir: str) -> str:
    """
    Fingerprint docs directory by file path + mtime + size.
    Changes when files are added/removed/modified.
    """
    parts: List[str] = []
    for p in iter_doc_files(docs_dir):
        try:
            st = os.stat(p)
            parts.append(f"{os.path.relpath(p, docs_dir)}|{int(st.st_mtime)}|{st.st_size}")
        except OSError:
            # If a file disappears mid-scan, skip it; next request will catch up.
            continue
    return "\n".join(parts)

