from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


SYSTEM_PROMPT = (
    "Eres el asistente oficial de la ETSI Informática. "
    "Responde de forma amable y breve usando SOLO el contexto proporcionado. "
    "Si no sabes la respuesta, dilo."
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
        # Inicializa el modelo de chat.
        self._llm = ChatGroq(
            model=model_name,
            temperature=temperature,
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Contexto relevante:\n\n{context}\n\n"
                    "Pregunta del usuario:\n{question}",
                ),
            ]
        )

        # Cadena simple: prompt -> LLM -> texto
        self._chain = self._prompt | self._llm | StrOutputParser()

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
        question: str,
        fragments: Sequence[Mapping[str, Any]],
    ) -> str:
        """
        Genera una respuesta final a partir de la pregunta del usuario y los
        fragmentos recuperados del vector store.

        - `question`: pregunta original del usuario.
        - `fragments`: lista de fragmentos, típicamente devueltos por VectorStoreManager.search().
        """
        if not question:
            raise ValueError("La pregunta no puede estar vacía.")

        if not fragments:
            # Forzamos el comportamiento de 'si no sabes la respuesta, dilo'
            return (
                "No dispongo de suficiente contexto para responder a esa pregunta. "
                "Prueba a reformularla o consulta con la secretaría de la ETSI Informática."
            )

        context = self._format_context(fragments)

        return self._chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )


__all__ = ["ChatEngine", "SYSTEM_PROMPT"]

