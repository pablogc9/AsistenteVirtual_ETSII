"""
Capa de recuperación sobre el grafo de conocimiento Neo4j.

Flujo por petición:
  1. _extract_entities(question)
       → El LLM identifica qué titulaciones, normativas o conceptos menciona
         la pregunta y decide si vale la pena consultar el grafo.

  2. _query_titulaciones(keywords) + _query_normativas(keywords)
       → Consultas Cypher CONTAINS que devuelven nodos con su estado de vigencia.

  3. _format_context(tit_rows, norm_rows)
       → Convierte los resultados en un bloque de texto estructurado que el LLM
         recibirá con indicación explícita de prioridad máxima.

El método público search(question) → str | None devuelve:
  - Un bloque de texto listo para inyectar en el prompt del LLM.
  - None si el grafo no tiene información relevante para esta pregunta
    (así no se contamina el contexto con basura).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel, Field

load_dotenv()

# ── Prompt de reescritura conversacional ──────────────────────────────────────

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

# ── Prompt de extracción de entidades ─────────────────────────────────────────

_ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un extractor de entidades para un sistema de recuperación sobre el grafo de conocimiento "
        "de la ETSI Informática (UMA).\n\n"
        "Tu tarea es analizar la pregunta del usuario y extraer:\n"
        "  1. Nombres o palabras clave de titulaciones mencionadas "
        "(grados, másteres, dobles grados, o títulos extintos como ITIS, ITIG, Licenciatura).\n"
        "  2. Palabras clave de normativa o conceptos académicos "
        "(TFG, TFM, Prácticum, prácticas externas, reconocimiento, permanencia...).\n"
        "  3. Si la pregunta realmente necesita consultar el grafo de conocimiento "
        "(False para preguntas sobre fechas de exámenes, horarios, reservas, etc.).\n\n"
        "Devuelve ÚNICAMENTE el JSON solicitado, sin texto adicional.",
    ),
    ("human", "Pregunta: {question}"),
])


class _GraphQuery(BaseModel):
    titulacion_keywords: list[str] = Field(
        default_factory=list,
        description="Palabras clave para buscar titulaciones en Neo4j (ej: ['Software', 'Computadores', 'ITIS']).",
    )
    normativa_keywords: list[str] = Field(
        default_factory=list,
        description="Palabras clave para buscar normativas o conceptos académicos (ej: ['TFG', 'Prácticum']).",
    )
    needs_graph: bool = Field(
        default=False,
        description="True si la pregunta involucra titulaciones, vigencia, normativa o planes de estudio.",
    )


# ── Consultas Cypher ──────────────────────────────────────────────────────────

_CYPHER_TITULACION = """
MATCH (t:Titulacion)-[:TIENE_PLAN]->(p:PlanEstudio)-[:TIENE_ESTADO]->(s:Estado)
WHERE any(kw IN $keywords WHERE toLower(t.nombre) CONTAINS toLower(kw)
                             OR toLower(t.tipo)   CONTAINS toLower(kw))
OPTIONAL MATCH (p)-[:TIENE_DISTRIBUCION]->(d:DistribucionCreditos)
OPTIONAL MATCH (t)-[:OFERTA_MENCION]->(m:Mencion WHERE m.activa_en_uma = true)
OPTIONAL MATCH (t)-[:IMPARTE_CON]->(u:Universidad)
RETURN
    t.nombre                AS titulo,
    t.tipo                  AS tipo,
    t.url                   AS url,
    p.nombre                AS plan,
    p.ects_total            AS ects,
    p.anios_duracion        AS anios,
    s.valor                 AS estado,
    s.descripcion           AS descripcion_estado,
    d.formacion_basica      AS fb,
    d.obligatorios          AS ob,
    d.optativos             AS op,
    d.practicas_externas    AS pe,
    d.tfg                   AS tfg,
    collect(DISTINCT m.nombre) AS menciones,
    collect(DISTINCT u.nombre) AS co_universidades
ORDER BY t.nombre, p.anio_inicio
"""

# Consulta para listar TODAS las titulaciones (sin filtro de keywords)
_CYPHER_ALL_TITULACIONES = """
MATCH (t:Titulacion)-[:TIENE_PLAN]->(p:PlanEstudio)-[:TIENE_ESTADO]->(s:Estado)
RETURN
    t.nombre    AS titulo,
    t.tipo      AS tipo,
    s.valor     AS estado,
    p.ects_total AS ects,
    p.anios_duracion AS anios
ORDER BY
    CASE s.valor
        WHEN 'Vigente'       THEN 1
        WHEN 'EnImplantacion' THEN 2
        WHEN 'EnExtincion'   THEN 3
        WHEN 'Extinto'       THEN 4
        ELSE 5
    END, t.nombre
"""

_CYPHER_NORMATIVA = """
MATCH (n:Normativa)
WHERE any(kw IN $keywords WHERE toLower(n.nombre) CONTAINS toLower(kw)
                             OR toLower(n.ambito)  CONTAINS toLower(kw)
                             OR toLower(coalesce(n.nota, '')) CONTAINS toLower(kw))
RETURN
    n.nombre                    AS nombre,
    n.tipo                      AS tipo,
    n.ambito                    AS ambito,
    n.nota                      AS nota,
    n.aplica_planes_vigentes    AS vigente,
    n.aplica_planes_extintos    AS extinto
"""


# ── Formateador de contexto ───────────────────────────────────────────────────

def _format_titulacion_row(row: dict[str, Any]) -> str:
    """Convierte una fila de Neo4j (titulación) en texto legible para el LLM."""
    lines: list[str] = []

    titulo = row.get("titulo", "Desconocido")
    tipo   = row.get("tipo", "")
    plan   = row.get("plan", "")
    estado = row.get("estado", "")
    desc   = row.get("descripcion_estado", "")
    ects   = row.get("ects")
    anios  = row.get("anios")

    lines.append(f"TITULACIÓN: {titulo} ({tipo})")
    lines.append(f"  PLAN: {plan}")
    lines.append(f"  ESTADO: {estado} — {desc}")

    if ects:
        lines.append(f"  CRÉDITOS TOTALES: {ects} ECTS en {anios} año(s)")

    # Distribución de créditos (si está disponible)
    fb = row.get("fb"); ob = row.get("ob"); op = row.get("op")
    pe = row.get("pe"); tfg = row.get("tfg")
    if all(v is not None for v in [fb, ob, op, pe, tfg]):
        lines.append(
            f"  DISTRIBUCIÓN: {fb} básica + {ob} obligatorios + {op} optativos "
            f"+ {pe} prácticas externas + {tfg} TFG"
        )

    menciones = [m for m in (row.get("menciones") or []) if m]
    if menciones:
        lines.append(f"  MENCIONES ACTIVAS EN UMA: {', '.join(menciones)}")

    co_unis = [u for u in (row.get("co_universidades") or []) if u]
    if co_unis:
        lines.append(f"  IMPARTIDO CONJUNTAMENTE CON: {', '.join(co_unis)}")

    url = row.get("url")
    if url:
        lines.append(f"  MÁS INFO: {url}")

    return "\n".join(lines)


def _format_normativa_row(row: dict[str, Any]) -> str:
    """Convierte una fila de Neo4j (normativa) en texto legible para el LLM."""
    lines: list[str] = []
    nombre = row.get("nombre", "Normativa desconocida")
    ambito = row.get("ambito", "")
    nota   = row.get("nota", "")
    v      = row.get("vigente")
    e      = row.get("extinto")

    aplicacion_parts: list[str] = []
    if v:
        aplicacion_parts.append("planes VIGENTES")
    if e:
        aplicacion_parts.append("planes EXTINTOS/ANTIGUOS")
    aplicacion = " y ".join(aplicacion_parts) if aplicacion_parts else "sin especificar"

    lines.append(f"NORMATIVA: {nombre} (ámbito: {ambito})")
    lines.append(f"  APLICA A: {aplicacion}")
    if nota:
        lines.append(f"  CONTENIDO CLAVE: {nota}")

    return "\n".join(lines)


def _build_context_block(
    tit_rows:  list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
) -> str | None:
    """
    Combina los resultados del grafo en un bloque de texto con marcador de prioridad.
    Devuelve None si no hay nada relevante.
    """
    sections: list[str] = []

    if tit_rows:
        formatted = [_format_titulacion_row(dict(r)) for r in tit_rows]
        sections.append("═══ TITULACIONES ═══\n" + "\n\n".join(formatted))

    if norm_rows:
        formatted = [_format_normativa_row(dict(r)) for r in norm_rows]
        sections.append("═══ NORMATIVA ═══\n" + "\n\n".join(formatted))

    if not sections:
        return None

    header = (
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  DATOS DEL GRAFO DE CONOCIMIENTO — PRIORIDAD MÁXIMA     ║\n"
        "║  Estos datos son ESTRUCTURADOS y VERIFICADOS.            ║\n"
        "║  Tienen PREFERENCIA sobre cualquier fragmento de texto.  ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )
    return header + "\n\n" + "\n\n".join(sections)


# ── Clase principal ───────────────────────────────────────────────────────────

class GraphRetriever:
    """Consultar Neo4j para enriquecer el contexto con titulaciones y normativa."""

    def __init__(
        self,
        uri:       str | None = None,
        user:      str | None = None,
        password:  str | None = None,
        llm_model: str        = "llama-3.1-8b-instant",
    ) -> None:
        self._uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = user     or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver:  Driver | None = None
        self._available: bool        = False

        self._connect()

        llm = ChatGroq(model=llm_model, temperature=0)
        self._entity_chain  = _ENTITY_PROMPT  | llm.with_structured_output(_GraphQuery)
        self._rewrite_chain = _REWRITE_PROMPT | llm

    # ── Conexión ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if not self._password:
            print("[GraphRetriever] NEO4J_PASSWORD no configurado — grafo deshabilitado.")
            return
        try:
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._driver.verify_connectivity()
            self._available = True
            print(f"[GraphRetriever] Conectado a Neo4j en {self._uri}.")
        except Exception as exc:
            print(f"[GraphRetriever] Neo4j no disponible ({exc}) — el RAG seguirá sin grafo.")
            self._driver    = None
            self._available = False

    def is_available(self) -> bool:
        return self._available

    # ── Reescritura conversacional ────────────────────────────────────────────

    def _rewrite_with_history(
        self,
        question: str,
        historial: list[dict[str, str]],
    ) -> str:
        """Reescribe la pregunta para que sea autocontenida usando el historial."""
        if not historial:
            return question
        try:
            history_text = "\n".join(
                f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                for m in historial[-4:]  # últimos 2 turnos (4 mensajes)
            )
            result = self._rewrite_chain.invoke({
                "history":  history_text,
                "question": question,
            })
            rewritten = result.content.strip() if hasattr(result, "content") else str(result).strip()
            return rewritten or question
        except Exception:
            return question

    # ── Extracción de entidades ───────────────────────────────────────────────

    def _extract_entities(self, question: str) -> _GraphQuery:
        """
        Usa el LLM para identificar si la pregunta necesita consultar el grafo
        y qué entidades buscar.

        Devuelve un fallback conservador si el LLM falla.
        """
        try:
            result = self._entity_chain.invoke({"question": question})
            return result  # type: ignore[return-value]
        except Exception:
            # Fallback por palabras clave hardcodeadas para los casos más frecuentes
            q_lower = question.lower()
            tit_kws: list[str] = []
            nor_kws: list[str] = []

            tit_hints = [
                "software", "informática", "computadores", "salud",
                "ciberseguridad", "inteligencia artificial", "matemáticas",
                "matematicas", "doble grado", "doble", "dgiim",
                "máster", "master", "doctorado",
                "itig", "itis", "licenciatura", "ingeniería técnica",
            ]
            nor_hints = [
                "tfg", "tfm", "prácticum", "prácticas externas",
                "reconocimiento", "permanencia", "reglamento",
                "extinción", "extinto", "plan antiguo",
            ]
            for hint in tit_hints:
                if hint in q_lower:
                    tit_kws.append(hint)
            for hint in nor_hints:
                if hint in q_lower:
                    nor_kws.append(hint)

            needs = bool(tit_kws or nor_kws)
            return _GraphQuery(
                titulacion_keywords=tit_kws,
                normativa_keywords=nor_kws,
                needs_graph=needs,
            )

    # ── Consultas Cypher ──────────────────────────────────────────────────────

    def _is_listing_all(self, question: str) -> bool:
        """Detecta si el usuario pide un listado completo de titulaciones."""
        q = question.lower()
        listing_hints = [
            "qué grados", "que grados", "cuáles son los grados", "cuales son los grados",
            "qué titulaciones", "que titulaciones", "cuáles son las titulaciones",
            "qué carreras", "que carreras", "cuáles son las carreras",
            "grados que", "grados hay", "titulaciones hay", "carreras hay",
            "lista de grados", "listado de grados", "todos los grados",
            "qué se imparte", "que se imparte", "qué ofrece", "que ofrece",
            "oferta académica", "oferta formativa",
            "todos los títulos", "todas las titulaciones",
        ]
        return any(hint in q for hint in listing_hints)

    def _query_all_titulaciones(self) -> list[Any]:
        """Devuelve todas las titulaciones sin filtro de keywords."""
        if not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(_CYPHER_ALL_TITULACIONES)
                return list(result)
        except Exception:
            return []

    def _format_listing(self, rows: list[Any]) -> str:
        """Formatea un listado completo de titulaciones de forma compacta.

        Deduplica por nombre de titulación: si hay varios planes del mismo grado
        se muestra solo uno (el más reciente, que llega primero por ORDER BY).
        """
        seen: set[str] = set()
        vigentes, extintos = [], []
        for row in rows:
            r = dict(row)
            titulo = r.get("titulo", "")
            if titulo in seen:
                continue  # ignorar planes adicionales del mismo grado
            seen.add(titulo)

            estado = r.get("estado", "")
            ects   = r.get("ects")
            anios  = r.get("anios")
            extra  = f" — {ects} ECTS, {anios} años" if ects and anios else ""
            line   = f"  • {titulo} ({r.get('tipo', '')}){extra}"
            if estado in ("Vigente", "EnImplantacion"):
                vigentes.append(line)
            elif estado == "EnExtincion":
                # Sigue existiendo (estudiantes matriculados), pero sin nueva admisión
                vigentes.append(f"{line}  [sin nueva admisión]")
            else:
                extintos.append(f"{line}  [EXTINTO]")

        parts: list[str] = []
        if vigentes:
            parts.append("TITULACIONES EN LA ETSII (UMA):\n" + "\n".join(vigentes))
        if extintos:
            parts.append("TITULACIONES YA EXTINTAS (planes antiguos):\n" + "\n".join(extintos))
        return "\n\n".join(parts)

    def _query_titulaciones(self, keywords: list[str]) -> list[Any]:
        if not keywords or not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(_CYPHER_TITULACION, keywords=keywords)
                return list(result)
        except Exception:
            return []

    def _query_normativas(self, keywords: list[str]) -> list[Any]:
        if not keywords or not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(_CYPHER_NORMATIVA, keywords=keywords)
                return list(result)
        except Exception:
            return []

    # ── Método público ────────────────────────────────────────────────────────

    def search(
        self,
        question:  str,
        historial: list[dict[str, str]] | None = None,
    ) -> str | None:
        """
        Analizar la pregunta, consultar Neo4j y devolver contexto estructurado.
        Devolver None si el grafo no está disponible o no aplica.
        """
        if not self._available:
            return None

        # 0. Reescribir la pregunta con contexto conversacional si hay historial
        effective_question = self._rewrite_with_history(question, historial or [])

        # Atajo: si la pregunta pide un listado de todas las titulaciones,
        # devolver directamente todas sin pasar por el extractor de keywords
        if self._is_listing_all(effective_question):
            all_rows = self._query_all_titulaciones()
            if not all_rows:
                return None
            header = (
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  DATOS DEL GRAFO DE CONOCIMIENTO — PRIORIDAD MÁXIMA     ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
            return header + "\n\n" + self._format_listing(all_rows)

        # 1. Extraer entidades con el LLM (sobre la pregunta reescrita)
        graph_query = self._extract_entities(effective_question)

        if not graph_query.needs_graph:
            return None

        # 2. Consultas Cypher
        tit_rows  = self._query_titulaciones(graph_query.titulacion_keywords)
        norm_rows = self._query_normativas(
            graph_query.normativa_keywords + graph_query.titulacion_keywords
        )

        # 3. Construir y devolver el bloque de contexto (o None si está vacío)
        return _build_context_block(tit_rows, norm_rows)

    def close(self) -> None:
        if self._driver:
            self._driver.close()


__all__ = ["GraphRetriever"]
