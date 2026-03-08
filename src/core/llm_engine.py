from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


SYSTEM_PROMPT = (
    "Eres el Asistente Virtual Oficial de la ETSI Informática de la Universidad de Málaga (UMA). "
    "Tu misión es responder preguntas basándote en el contexto proporcionado junto con los DATOS MAESTROS que se indican a continuación.\n\n"

    "DATOS MAESTROS — son hechos oficiales verificados que debes afirmar con total seguridad y sin ningún disclaimer:\n"
    "- El Grado en Ingeniería del Software tiene exactamente 240 créditos ECTS en total.\n"
    "- El Grado en Ingeniería Informática tiene exactamente 240 créditos ECTS en total.\n"
    "- Ambos grados tienen una duración de 4 años (8 semestres).\n"
    "- Se imparten en la Escuela Técnica Superior de Ingeniería Informática de la UMA.\n\n"

    "REGLAS CRÍTICAS DE COMPORTAMIENTO:\n"
    "1. Los DATOS MAESTROS anteriores son HECHOS ABSOLUTOS. Cuando te pregunten por el total de créditos de un grado, "
    "responde directamente con el número exacto SIN añadir frases como 'no se especifica explícitamente' o 'se puede inferir'. "
    "Ejemplo correcto: 'El Grado en Ingeniería del Software tiene **240 créditos ECTS**.' "
    "Ejemplo INCORRECTO: 'Aunque no se especifica explícitamente...'\n"
    "2. Si la respuesta no está en el contexto ni en los DATOS MAESTROS, di: 'Lo siento, no tengo información oficial sobre eso en mis registros.'\n"
    "3. PROHIBIDO dar consejos personales, opiniones o sugerencias externas.\n"
    "4. No inventes datos que no estén en el contexto o los DATOS MAESTROS.\n"
    "5. Mantén un tono profesional, institucional y conciso.\n"
    "6. Si el usuario te pregunta quién eres, responde que eres el asistente oficial de la ETSI Informática.\n"
    "7. Sé preciso con las cifras: distingue entre créditos de módulos individuales y el total del grado.\n"
    "8. Usa Markdown para estructurar tus respuestas: negritas para datos clave, listas para enumeraciones."
)


class ChatEngine:
    """
    Motor de chat basado en LangChain + Llama-3 (Groq).

    - Recibe una pregunta del usuario y fragmentos recuperados de ChromaDB.
    - Construye un mensaje con el contexto y genera una respuesta final.
    """

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
        """
        Recibe los fragmentos recuperados de ChromaDB y los empaqueta en un texto
        que el modelo pueda usar como contexto.

        Se asume que cada fragmento tiene al menos:
        - `text`: contenido del chunk.
        - `metadata`: diccionario con metadatos (título, URL, etc.).
        """
        lines: list[str] = []

        for i, frag in enumerate(fragments, start=1):
            text = frag.get("text", "")
            metadata = frag.get("metadata", {}) or {}
            title = metadata.get("title") or metadata.get("Titulo") or ""
            source = metadata.get("source") or metadata.get("url") or ""

            header_parts: list[str] = []
            if title:
                header_parts.append(f"Título: {title}")
            if source:
                header_parts.append(f"Fuente: {source}")

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
    ) -> tuple[str, int]:
        """
        Genera una respuesta final a partir de la pregunta del usuario y los
        fragmentos recuperados del vector store.

        - `system_prompt`: si se proporciona, sobreescribe SYSTEM_PROMPT (configurable desde el admin).
        - `model_name`: si se proporciona y es distinto al por defecto, crea un ChatGroq al vuelo.
        """
        if not question:
            raise ValueError("La pregunta no puede estar vacía.")

        if not fragments:
            return (
                "No dispongo de suficiente contexto para responder a esa pregunta. "
                "Prueba a reformularla o consulta con la secretaría de la ETSI Informática.",
                0,
            )

        active_prompt = system_prompt or SYSTEM_PROMPT

        # Reutilizar el LLM base salvo que el admin haya cambiado el modelo
        if model_name and model_name != self._model_name:
            active_llm = ChatGroq(model=model_name, temperature=self._temperature)
        else:
            active_llm = self._llm

        context = self._format_context(fragments)

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

