from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from src.database.vector_store import VectorStoreManager


# ── Prompt de generación de variaciones ──────────────────────────────────────

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un experto en búsqueda semántica para un sistema RAG académico.\n"
        "Genera exactamente 3 reformulaciones de la pregunta del usuario que "
        "maximicen la cobertura semántica en una búsqueda vectorial.\n\n"
        "Cada reformulación debe:\n"
        "  - Usar vocabulario alternativo o sinónimos\n"
        "  - Abordar el mismo tema desde un ángulo diferente\n"
        "  - Ser útil para recuperar fragmentos distintos pero relevantes\n\n"
        "Responde ÚNICAMENTE con las 3 reformulaciones, una por línea, "
        "sin numeración, viñetas ni explicaciones adicionales.",
    ),
    ("human", "{question}"),
])

# Umbral de distancia vectorial para filtrar candidatos antes del re-ranking.
# Un valor más permisivo (0.75) asegura que el reranker multilingüe reciba
# suficientes candidatos incluso para queries cortas o ambiguas.
_DISTANCE_THRESHOLD = 0.75


class AdvancedRetriever:
    """
    Pipeline de recuperación avanzado: Multi-Query + Re-ranking con CrossEncoder.

    Flujo:
        1. Genera 3 variaciones de la pregunta con el LLM (Multi-Query).
        2. Busca en ChromaDB con la pregunta original + las 3 variaciones.
        3. Consolida y deduplica los candidatos (hasta 4 × candidates_per_query).
        4. Re-rankea con un CrossEncoder ligero y devuelve los Top-K finales.
    """

    def __init__(
        self,
        vector_store:         VectorStoreManager,
        llm_model:            str = "llama-3.1-8b-instant",
        candidates_per_query: int = 12,
        top_k:                int = 5,
        reranker_model:       str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ) -> None:
        self._vector_store         = vector_store
        self._candidates_per_query = candidates_per_query
        self._top_k                = top_k

        llm = ChatGroq(model=llm_model, temperature=0)
        self._query_chain = _MULTI_QUERY_PROMPT | llm
        self._reranker    = CrossEncoder(reranker_model)

    # ── Paso 1: generación de variaciones ────────────────────────────────────

    def _generate_variations(
        self, 
        question: str,
    ) -> list[str]:
        """Devuelve hasta 3 reformulaciones semánticas de la pregunta."""
        try:
            response = self._query_chain.invoke({"question": question})
            lines = [
                line.strip()
                for line in response.content.strip().splitlines()
                if line.strip()
            ]
            return lines[:3]
        except Exception:
            return []

    # ── Paso 2: búsqueda multi-query y deduplicación ─────────────────────────

    def _search_and_deduplicate(
        self, 
        queries: list[str]
    ) -> list[dict[str, Any]]:
        """
        Ejecuta una búsqueda vectorial para cada query y consolida los resultados.

        La deduplicación usa SHA-256 del texto (idéntico al ID interno de ChromaDB).
        Los candidatos con distancia vectorial > _DISTANCE_THRESHOLD se descartan
        antes de llegar al re-ranker, ahorrando tiempo de inferencia.
        """
        seen:       set[str]             = set()
        candidates: list[dict[str, Any]] = []

        for query in queries:
            results = self._vector_store.search(query, k=self._candidates_per_query)
            for result in results:
                if result.get("distance", 1.0) > _DISTANCE_THRESHOLD:
                    continue
                doc_id = hashlib.sha256(
                    result["text"].encode("utf-8")
                ).hexdigest()
                if doc_id not in seen:
                    seen.add(doc_id)
                    candidates.append(result)

        return candidates

    # ── Paso 3: re-ranking con CrossEncoder ──────────────────────────────────

    def _rerank(
        self, 
        question: str, 
        candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Puntúa cada candidato con el CrossEncoder y devuelve el Top-K.

        El CrossEncoder evalúa el par (pregunta, fragmento) como un único
        input, lo que produce puntuaciones de relevancia mucho más precisas
        que la similitud coseno.

        Si el re-ranking falla, hace fallback al orden de distancia vectorial.
        """
        if not candidates:
            return []

        try:
            pairs  = [(question, c["text"]) for c in candidates]
            scores = self._reranker.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            ranked = sorted(
                candidates, key=lambda x: x["rerank_score"], reverse=True
            )
            return ranked[: self._top_k]

        except Exception:
            # Fallback: ordena por distancia vectorial ascendente
            for c in candidates:
                c["rerank_score"] = 0.0
            sorted_by_dist = sorted(
                candidates, key=lambda x: x.get("distance", 1.0)
            )
            return sorted_by_dist[: self._top_k]

    # ── Método principal ──────────────────────────────────────────────────────

    def retrieve(
        self, 
        question: str
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Orquesta el pipeline completo.

        Returns:
            fragments  – Top-K fragmentos re-rankeados listos para el LLM.
            best_score – Puntuación CrossEncoder del fragmento más relevante
                         (útil para auditar calidad en el dashboard).
        """
        variations  = self._generate_variations(question)
        all_queries = [question] + variations

        candidates = self._search_and_deduplicate(all_queries)
        if not candidates:
            return [], 0.0

        top        = self._rerank(question, candidates)
        best_score = top[0]["rerank_score"] if top else 0.0

        return top, best_score


__all__ = ["AdvancedRetriever"]
