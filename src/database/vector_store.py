from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import chromadb
from chromadb.utils import embedding_functions

try:
    # Tipo de documento compatible con el procesador basado en LangChain
    from langchain_core.documents import Document as LCDocument
except ImportError:  # pragma: no cover - fallback cuando langchain_core no está disponible
    LCDocument = Any  # type: ignore


class VectorStoreManager:
    """Gestor de ChromaDB con persistencia en disco y embeddings multilingües."""

    def __init__(
        self,
        collection_name: str = "etsi_documents",
        persist_directory: str | None = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_dir = project_root / "data" / "chroma_db"

        self.persist_directory = Path(persist_directory) if persist_directory else default_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    def add_documents(
        self,
        chunks: Iterable[LCDocument],
        metadatas: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        """Añadir chunks al almacén vectorial con upsert idempotente por hash de contenido."""
        texts: List[str] = []
        metas: List[Mapping[str, Any]] = []

        provided_metas: List[Mapping[str, Any]] | None = list(metadatas) if metadatas is not None else None

        for idx, chunk in enumerate(chunks):
            content = getattr(chunk, "page_content", None)
            if not isinstance(content, str):
                raise ValueError("Cada chunk debe tener un atributo `page_content` de tipo str.")

            texts.append(content)

            if provided_metas is not None:
                if idx >= len(provided_metas):
                    raise ValueError("El número de metadatos no coincide con el número de chunks.")
                metas.append(provided_metas[idx])
            else:
                chunk_meta = getattr(chunk, "metadata", {}) or {}
                metas.append(dict(chunk_meta))

        if not texts:
            return

        ids: List[str] = [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ]

        unique_ids: List[str] = []
        unique_texts: List[str] = []
        unique_metas: List[Mapping[str, Any]] = []
        seen: set[str] = set()

        for doc_id, text, meta in zip(ids, texts, metas):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique_ids.append(doc_id)
            unique_texts.append(text)
            unique_metas.append(meta)

        self._collection.upsert(
            ids=unique_ids,
            documents=unique_texts,
            metadatas=list(unique_metas),
        )

    def search(self, query: str, k: int = 3) -> List[dict[str, Any]]:
        """Devolver los k fragmentos más similares a la consulta."""
        if not query:
            return []

        count = self._collection.count()
        if count == 0:
            return []
        k = min(k, count)

        result = self._collection.query(
            query_texts=[query],
            n_results=k,
        )

        documents: Sequence[str] = result.get("documents", [[]])[0]
        metadatas: Sequence[Mapping[str, Any]] = result.get("metadatas", [[]])[0]
        distances: Sequence[float] = result.get("distances", [[]])[0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]


__all__ = ["VectorStoreManager"]

