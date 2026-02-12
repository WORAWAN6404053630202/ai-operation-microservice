# code/service/local_vector_store.py
from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import conf


def _stringify_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in (metadata or {}).items():
        clean[k] = "" if v is None else str(v)
    return clean


class LocalVectorStoreManager:
    """
    Local-only VectorStore manager using Chroma
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

        self.vectorstore = Chroma(
            collection_name=conf.COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=self._persist_dir(),
        )

        count = self._collection_count()
        print(f"[VectorStore] Connected (collection={conf.COLLECTION_NAME})")
        print(f"[VectorStore] Collection count = {count}")

        if fail_if_empty and (count is None or count == 0):
            raise RuntimeError(
                "Local Chroma collection is empty. You must ingest documents first.\n"
                "Fix: run `python scripts/ingest_local.py` (or call ingest_documents())."
            )

        return self._build_retriever()

    def create_vectorstore(self, documents: List[Document]):
        self.initialize_embeddings()

        docs = [
            Document(
                page_content=d.page_content,
                metadata=_stringify_metadata(getattr(d, "metadata", {}) or {}),
            )
            for d in documents
        ]

        print(f"[VectorStore] Creating local Chroma ({len(docs)} docs)...")

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=conf.COLLECTION_NAME,
            persist_directory=self._persist_dir(),
        )

        try:
            self.vectorstore.persist()
        except Exception:
            pass

        count = self._collection_count()
        print(f"[VectorStore] Created successfully | count={count}")

        return self._build_retriever()

    def retrieve_docs(self, query: str) -> List[Dict]:
        if not self.retriever:
            raise RuntimeError("Retriever not initialized yet.")
        docs = self.retriever.invoke(query)
        return [{"content": doc.page_content[:600], "metadata": doc.metadata or {}} for doc in docs]


_MANAGER = LocalVectorStoreManager()


def get_retriever(k: int = 0, fail_if_empty: bool = True):
    """
    Public API for local usage.
    """
    if _MANAGER.retriever is not None:
        if k and int(k) > 0 and _MANAGER.vectorstore is not None:
            _MANAGER._build_retriever(k=int(k))
        return _MANAGER.retriever

    _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)

    if k and int(k) > 0:
        _MANAGER._build_retriever(k=int(k))

    return _MANAGER.retriever


def ingest_documents(documents: List[Document]):
    """Public API for ingestion."""
    return _MANAGER.create_vectorstore(documents)
