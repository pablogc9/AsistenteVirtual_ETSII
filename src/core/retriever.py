from __future__ import annotations

import hashlib
from typing import Any, Mapping

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from src.database.vector_store import VectorStoreManager


# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Dado el historial de conversación y la nueva pregunta del usuario, "
        "reescribe la pregunta para que sea completamente autocontenida: "
        "sustituye pronombres ambiguos ('ese', 'el otro', 'ambos', 'eso', etc.) "
        "por sus referentes explícitos tomados del historial.\n"
        "Si la pregunta ya es clara por sí sola, devuélvela sin modificar.\n"
        "Responde ÚNICAMENTE con la pregunta reescrita, sin explicaciones.",
    ),
    ("human", "Historial reciente:\n{history}\n\nNueva pregunta: {question}"),
])

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

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un asistente experto en la ETSI Informática de la Universidad de Málaga.\n"
        "Genera un fragmento breve (2-3 frases) que parezca extraído de un documento "
        "oficial de la ETSI y que respondería directamente a la pregunta del usuario.\n"
        "No uses frases como 'según los documentos' ni meta-referencias: escribe "
        "directamente como si fuera el texto del documento.\n"
        "Responde ÚNICAMENTE con el fragmento, sin explicaciones adicionales.",
    ),
    ("human", "{question}"),
])

# ── Constantes ────────────────────────────────────────────────────────────────

# Máxima distancia coseno permitida para llegar al re-ranker.
_DISTANCE_THRESHOLD = 0.75

# Score mínimo del CrossEncoder para que un fragmento se envíe al LLM.
# Valores típicos del modelo mmarco: relevante > 0, irrelevante < -3.
_MIN_RERANK_SCORE = -2.0


class AdvancedRetriever:
    """
    Pipeline de recuperación avanzado con cuatro mejoras sobre la búsqueda
    vectorial simple:

    1. Reescritura conversacional  – desambigua la query usando el historial
                                     antes de buscar (ej. 'el otro grado' →
                                     'el Grado en Ingeniería Informática').
    2. Multi-Query                 – genera 3 variaciones semánticas con el LLM
                                     para ampliar la cobertura de recuperación.
    3. HyDE                        – genera un documento hipotético que sirve
                                     como query adicional en espacio de embeddings
                                     de respuestas (no de preguntas).
    4. Re-ranking con CrossEncoder – puntúa cada candidato con un modelo
                                     multilingüe y filtra los irrelevantes
                                     (score < _MIN_RERANK_SCORE).
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
        self._rewrite_chain = _REWRITE_PROMPT     | llm
        self._query_chain   = _MULTI_QUERY_PROMPT | llm
        self._hyde_chain    = _HYDE_PROMPT         | llm
        self._reranker      = CrossEncoder(reranker_model)

    # ── Paso 0: reescritura conversacional ───────────────────────────────────

    def _rewrite_with_history(
        self,
        question: str,
        historial: list[Mapping[str, str]],
    ) -> str:
        """
        Si hay historial, reescribe la pregunta para que sea autocontenida.
        Usa solo los últimos 6 turnos (3 intercambios) para no sobrecargar el prompt.
        """
        if not historial:
            return question

        recent = historial[-6:]
        history_text = "\n".join(
            f"{'Usuario' if m.get('role') in ('human', 'user') else 'Asistente'}: {m.get('content', '')}"
            for m in recent
        )
        try:
            response = self._rewrite_chain.invoke({
                "history":  history_text,
                "question": question,
            })
            rewritten = response.content.strip()
            return rewritten if rewritten else question
        except Exception:
            return question

    # ── Paso 1a: variaciones Multi-Query ─────────────────────────────────────

    def _generate_variations(self, question: str) -> list[str]:
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

    # ── Paso 1b: documento hipotético (HyDE) ─────────────────────────────────

    def _generate_hyde(self, question: str) -> str:
        """
        Genera un fragmento de documento hipotético que respondería la pregunta.

        Por qué funciona: buscar con un texto que parece una *respuesta* produce
        embeddings más próximos a los documentos reales que buscar con la pregunta.
        """
        try:
            response = self._hyde_chain.invoke({"question": question})
            return response.content.strip()
        except Exception:
            return ""

    # ── Paso 2: búsqueda multi-query y deduplicación ─────────────────────────

    def _search_and_deduplicate(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        Ejecuta una búsqueda vectorial por cada query y consolida los resultados.

        Deduplicación por SHA-256 del texto; filtra candidatos con distancia > umbral.
        """
        seen:       set[str]             = set()
        candidates: list[dict[str, Any]] = []

        for query in queries:
            results = self._vector_store.search(query, k=self._candidates_per_query)
            for result in results:
                if result.get("distance", 1.0) > _DISTANCE_THRESHOLD:
                    continue
                doc_id = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
                if doc_id not in seen:
                    seen.add(doc_id)
                    candidates.append(result)

        return candidates

    # ── Paso 3: re-ranking con CrossEncoder ──────────────────────────────────

    def _rerank(
        self,
        question:   str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Puntúa cada candidato con el CrossEncoder y devuelve el Top-K filtrado.

        Solo se devuelven fragmentos con rerank_score >= _MIN_RERANK_SCORE,
        evitando que contexto claramente irrelevante llegue al LLM.
        Si ninguno supera el umbral, devuelve lista vacía → el sistema
        responde 'no tengo información'.
        """
        if not candidates:
            return []

        try:
            pairs  = [(question, c["text"]) for c in candidates]
            scores = self._reranker.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

            # Filtrar por umbral de relevancia mínima
            filtered = [c for c in ranked[: self._top_k] if c["rerank_score"] >= _MIN_RERANK_SCORE]
            return filtered

        except Exception:
            # Fallback: ordena por distancia vectorial ascendente sin filtro de score
            for c in candidates:
                c["rerank_score"] = 0.0
            return sorted(candidates, key=lambda x: x.get("distance", 1.0))[: self._top_k]

    # ── Método principal ──────────────────────────────────────────────────────

    def retrieve(
        self,
        question:  str,
        historial: list[Mapping[str, str]] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Orquesta el pipeline completo de recuperación.

        Parámetros:
            question  – Pregunta original del usuario.
            historial – Historial de la conversación (para reescritura contextual).

        Devuelve:
            fragments  – Top-K fragmentos relevantes listos para el LLM.
            best_score – Score CrossEncoder del mejor fragmento (para auditoría).
        """
        # 0. Reescritura conversacional (solo si hay historial)
        standalone = self._rewrite_with_history(question, historial or [])

        # 1. Multi-Query + HyDE → pool de queries diversas
        variations = self._generate_variations(standalone)
        hyde       = self._generate_hyde(standalone)

        all_queries = [standalone] + variations
        if hyde:
            all_queries.append(hyde)

        # 2. Búsqueda vectorial + deduplicación
        candidates = self._search_and_deduplicate(all_queries)
        if not candidates:
            return [], 0.0

        # 3. Re-ranking con filtro de score mínimo
        # El CrossEncoder evalúa relevancia con la pregunta standalone (más específica)
        top        = self._rerank(standalone, candidates)
        best_score = top[0]["rerank_score"] if top else 0.0

        return top, best_score


__all__ = ["AdvancedRetriever"]
