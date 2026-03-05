# code/service/local_vector_store.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import conf


# Safe directories for rmtree — prevents accidental deletion outside project scope
_SAFE_RMTREE_PARENTS = [
    str(Path.cwd()),
    str(Path.home() / ".cache"),
    str(Path.home() / "Downloads"),
]


def _safe_rmtree(path: Path) -> None:
    """
    Safety-wrapped shutil.rmtree.
    Validates that path is inside a known-safe parent directory before deleting.
    Raises RuntimeError if path is outside safe dirs instead of silently deleting.
    """
    resolved = str(path.resolve())
    if not any(resolved.startswith(safe) for safe in _SAFE_RMTREE_PARENTS):
        raise RuntimeError(
            f"[VectorStore] Refusing to delete '{resolved}' — path is outside safe directories. "
            f"Check LOCAL_VECTOR_DIR in env.properties."
        )
    shutil.rmtree(path)  # raise on error (no ignore_errors)


def _stringify_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in (metadata or {}).items():
        clean[k] = "" if v is None else str(v)
    return clean


class LocalVectorStoreManager:
    """
    Local-only VectorStore manager using Chroma

    NOTE (production boundary):
    - infra only; no policy
    """

    def __init__(self):
        self.embedding_model = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None

    def initialize_embeddings(self) -> None:
        if self.embedding_model is not None:
            return
        print(f"[Embedding] Loading: {conf.EMBEDDING_MODEL}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=conf.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embedding] Loaded successfully")

    def _persist_dir(self) -> str:
        base = Path(getattr(conf, "LOCAL_VECTOR_DIR", "./local_chroma"))
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    def _collection_name(self) -> str:
        return str(getattr(conf, "COLLECTION_NAME", "default_collection"))

    def _collection_count(self) -> Optional[int]:
        if not self.vectorstore:
            return None
        try:
            return int(self.vectorstore._collection.count())
        except Exception:
            return None

    def _build_retriever(self, k: Optional[int] = None):
        kk = int(k or getattr(conf, "RETRIEVAL_TOP_K", 20))
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": kk})
        print(f"[Retriever] Ready (k={kk})")
        return self.retriever

    def connect_to_existing(self, fail_if_empty: bool = True):
        print("[VectorStore] Connecting to local Chroma...")
        self.initialize_embeddings()

        persist_dir = self._persist_dir()
        collection_name = self._collection_name()

        print(f"[VectorStore] persist_directory = {Path(persist_dir).resolve()}")
        print(f"[VectorStore] collection_name   = {collection_name}")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_dir,
        )

        count = self._collection_count()
        print(f"[VectorStore] Connected (collection={collection_name})")
        print(f"[VectorStore] Collection count = {count}")

        if fail_if_empty and (count is None or count == 0):
            raise RuntimeError(
                "Local Chroma collection is empty. You must ingest documents first.\n"
                "Fix: run `PYTHONPATH=\"$PWD\" python -m code.scripts.ingest_local`"
            )

        return self._build_retriever()

    def create_vectorstore(self, documents: List[Document], reset: bool = True):
        """
        Build a fresh local Chroma vectorstore from documents.

        reset=True:
          - delete persist_directory to avoid duplicate/old vectors
          - guarantees stable count after ingest
        """
        self.initialize_embeddings()

        persist_dir = self._persist_dir()
        collection_name = self._collection_name()

        if reset:
            # IMPORTANT: wipe the actual persist directory used by the app
            p = Path(persist_dir)
            if p.exists():
                print(f"[VectorStore] Reset enabled -> removing persist dir: {p.resolve()}")
                _safe_rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        docs = [
            Document(
                page_content=d.page_content,
                metadata=_stringify_metadata(getattr(d, "metadata", {}) or {}),
            )
            for d in (documents or [])
        ]

        print(f"[VectorStore] Creating local Chroma ({len(docs)} docs)...")
        print(f"[VectorStore] persist_directory = {Path(persist_dir).resolve()}")
        print(f"[VectorStore] collection_name   = {collection_name}")

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        try:
            self.vectorstore.persist()
        except Exception:
            pass

        count = self._collection_count()
        print(f"[VectorStore] Created successfully | count={count}")

        return self._build_retriever()

    # ------------------------------------------------------------------
    # Retrieval helpers (infra-only)
    # ------------------------------------------------------------------
    def retrieve_raw_docs(self, query: str, k: Optional[int] = None) -> List[Document]:
        if not query or not str(query).strip():
            return []
        if not self.retriever:
            raise RuntimeError("Retriever not initialized yet.")

        if k and int(k) > 0 and self.vectorstore is not None:
            tmp = self.vectorstore.as_retriever(search_kwargs={"k": int(k)})
            docs = tmp.invoke(query)
        else:
            docs = self.retriever.invoke(query)

        return list(docs or [])

    def retrieve_docs(self, query: str, k: Optional[int] = None, clip_chars: int = 600) -> List[Dict]:
        docs = self.retrieve_raw_docs(query, k=k)
        out: List[Dict] = []
        for doc in docs:
            out.append(
                {
                    "content": (getattr(doc, "page_content", "") or "")[: int(clip_chars or 600)],
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        return out


_MANAGER = LocalVectorStoreManager()


def get_retriever(k: int = 0, fail_if_empty: bool = True):
    if _MANAGER.retriever is not None:
        if k and int(k) > 0 and _MANAGER.vectorstore is not None:
            _MANAGER._build_retriever(k=int(k))
        return _MANAGER.retriever

    _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)

    if k and int(k) > 0:
        _MANAGER._build_retriever(k=int(k))

    return _MANAGER.retriever


def retrieve_raw_docs(query: str, k: int = 0, fail_if_empty: bool = True) -> List[Document]:
    if _MANAGER.retriever is None:
        _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)
    kk = int(k) if k and int(k) > 0 else None
    return _MANAGER.retrieve_raw_docs(query, k=kk)


def retrieve_docs(query: str, k: int = 0, clip_chars: int = 600, fail_if_empty: bool = True) -> List[Dict]:
    if _MANAGER.retriever is None:
        _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)
    kk = int(k) if k and int(k) > 0 else None
    return _MANAGER.retrieve_docs(query, k=kk, clip_chars=clip_chars)


def ingest_documents(documents: List[Document], reset: bool = True):
    """Public API for ingestion."""
    return _MANAGER.create_vectorstore(documents, reset=reset)