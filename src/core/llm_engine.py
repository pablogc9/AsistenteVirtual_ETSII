from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


SYSTEM_PROMPT = (
    "Eres el Asistente Virtual Oficial de la ETSI Informática de la Universidad de Málaga (UMA). "
    "Tu misión es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado.\n\n"

    "FORMATO DEL CONTEXTO:\n"
    "Los fragmentos de contexto provienen de documentos PDF convertidos a Markdown. Esto implica:\n"
    "- Las TABLAS están en formato Markdown (| col | col |). Léelas fila a fila para extraer datos precisos.\n"
    "- Los TÍTULOS de sección (# ## ###) indican la jerarquía del documento. El campo 'Sección' de cada "
    "fragmento muestra la ruta completa (ej: 'Capítulo I > Artículo 3'). Úsala para situar la información.\n"
    "- Si el mismo dato aparece en una tabla y en el texto, la tabla tiene prioridad.\n\n"

    "DATOS MAESTROS — hechos oficiales verificados, afírmalos sin ningún disclaimer:\n"
    "- El Grado en Ingeniería del Software tiene exactamente 240 créditos ECTS en total.\n"
    "- El Grado en Ingeniería Informática tiene exactamente 240 créditos ECTS en total.\n"
    "- Ambos grados tienen una duración de 4 años (8 semestres).\n"
    "- Se imparten en la Escuela Técnica Superior de Ingeniería Informática de la UMA.\n\n"

    "REGLAS CRÍTICAS DE COMPORTAMIENTO:\n"
    "1. Los DATOS MAESTROS son HECHOS ABSOLUTOS. Responde directamente con el número exacto SIN añadir "
    "frases como 'no se especifica explícitamente' o 'se puede inferir'.\n"
    "2. NUNCA cites el nombre de documentos ni secciones en tu respuesta (nada de 'según el documento X' "
    "ni 'en la sección Y'). Las fuentes se muestran aparte; tu respuesta debe ser directa y natural.\n"
    "3. Si la respuesta está en una tabla del contexto, extrae los datos concretos de esa tabla.\n"
    "4. Si la respuesta no está en el contexto ni en los DATOS MAESTROS, di: "
    "'Lo siento, no tengo información oficial sobre eso en mis registros.'\n"
    "5. PROHIBIDO dar consejos personales, opiniones o sugerencias externas.\n"
    "6. No inventes datos que no estén en el contexto o los DATOS MAESTROS.\n"
    "7. Mantén un tono profesional, institucional y conciso.\n"
    "8. Usa Markdown para estructurar tus respuestas: negritas para datos clave, listas para enumeraciones.\n"
    "9. Sé preciso con las cifras: distingue entre créditos de módulos individuales y el total del grado.\n"
    "10. NUNCA menciones de dónde proviene la información interna (grafo, fragmentos, ChromaDB, etc.). "
    "Responde directamente como si el conocimiento fuera tuyo. En lugar de 'Según el grafo...' di simplemente los datos."
)


class ChatEngine:
    """Generar respuestas a partir de fragmentos vectoriales y contexto del grafo."""

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._llm = ChatGroq(model=model_name, temperature=temperature)


    @staticmethod
    def _format_context(
        fragments: Iterable[Mapping[str, Any]],
    ) -> str:
        """Formatear fragmentos de ChromaDB con metadatos de documento y sección."""
        lines: list[str] = []

        for i, frag in enumerate(fragments, start=1):
            text     = frag.get("text", "")
            metadata = frag.get("metadata", {}) or {}

            # ── Título del documento ───────────────────────────────────────────
            title = (
                metadata.get("title")
                or metadata.get("Titulo")
                or ""
            )

            # ── Fuente (nombre de fichero o URL) ───────────────────────────────
            source = (
                metadata.get("source_file")
                or metadata.get("source")
                or metadata.get("url")
                or ""
            )

            # ── Sección jerárquica (nuevo metadato de etsi_hibrida) ────────────
            section = metadata.get("section", "").strip()

            # ── Construcción del encabezado del fragmento ──────────────────────
            header_parts: list[str] = []
            if title:
                header_parts.append(f"Documento: {title}")
            if source:
                header_parts.append(f"Fuente: {source}")
            if section:
                header_parts.append(f"Sección: {section}")

            header = " | ".join(header_parts) if header_parts else f"Fragmento {i}"

            lines.append(f"[{header}]\n{text}")

        return "\n\n---\n\n".join(lines)

    def generate_answer(
        self,
        question:      str,
        fragments:     Sequence[Mapping[str, Any]],
        historial:     list[Mapping[str, Any]],
        system_prompt: str | None = None,
        model_name:    str | None = None,
        graph_context: str | None = None,
    ) -> tuple[str, int]:
        """Generar respuesta final combinando grafo, fragmentos e historial."""
        if not question:
            raise ValueError("La pregunta no puede estar vacía.")

        if not fragments and not graph_context:
            return (
                "No dispongo de suficiente contexto para responder a esa pregunta. "
                "Prueba a reformularla o consulta con la secretaría de la ETSI Informática.",
                0,
            )

        active_prompt = system_prompt or SYSTEM_PROMPT

        if model_name and model_name != self._model_name:
            active_llm = ChatGroq(model=model_name, temperature=self._temperature)
        else:
            active_llm = self._llm

        context_parts: list[str] = []
        if graph_context:
            context_parts.append(graph_context)
        if fragments:
            chroma_block = self._format_context(fragments)
            header = (
                "── FRAGMENTOS DE DOCUMENTOS (contexto de apoyo) ──────────────\n"
                "Usa estos fragmentos para complementar la respuesta, pero en caso\n"
                "de contradicción, los DATOS DEL GRAFO tienen prioridad absoluta.\n"
                "───────────────────────────────────────────────────────────────"
            )
            context_parts.append(f"{header}\n\n{chroma_block}")

        context = "\n\n".join(context_parts)

        messages = [("system", active_prompt)]
        for msg in historial:
            messages.append((
                msg.role    if hasattr(msg, "role")    else msg["role"],
                msg.content if hasattr(msg, "content") else msg["content"],
            ))
        messages.append(("human", f"Contexto relevante:\n\n{context}\n\nPregunta del usuario:\n{question}"))

        prompt      = ChatPromptTemplate.from_messages(messages)
        ai_message  = (prompt | active_llm).invoke({})

        answer      = ai_message.content
        usage       = ai_message.response_metadata.get("token_usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return answer, tokens_used


__all__ = ["ChatEngine", "SYSTEM_PROMPT"]

