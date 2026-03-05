from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


SYSTEM_PROMPT = (
    "Eres el Asistente Virtual Oficial de la ETSI Informática de la Universidad de Málaga (UMA). "
    "Tu única misión es responder preguntas basadas EXCLUSIVAMENTE en el contexto proporcionado. "
    "Tu misión es responder basándote en el contexto, pero tienes estos DATOS MAESTROS que son siempre ciertos:\n"
    "1. El Grado en Ingeniería del Software tiene un total de 240 créditos ECTS.\n"
    "2. La duración es de 4 años.\n"
    "3. Se imparte en la Escuela Técnica Superior de Ingeniería Informática.\n\n"
    
    "REGLAS CRÍTICAS DE COMPORTAMIENTO:\n"
    "1. Si la respuesta no está en el contexto, di exactamente: 'Lo siento, no tengo información oficial sobre eso en mis registros.'\n"
    "2. PROHIBIDO dar consejos personales, opiniones o sugerencias externas (como 'busca en Google' o 'pregunta a compañeros').\n"
    "3. No inventes datos. Si el contexto habla de la feria de empleo y te preguntan por pizzas, aplica la Regla 1.\n"
    "4. Mantén un tono profesional, institucional y conciso.\n"
    "5. Si el usuario te pregunta quién eres, responde que eres el asistente oficial de la escuela.\n"
    "6. Sé extremadamente preciso con las cifras. Si en un fragmento aparece que el grado tiene 240 créditos y en otro habla de 12 créditos de prácticas, distingue claramente entre ambos.\n"
    "7. Antes de afirmar que un dato 'no se especifica', asegúrate de haber leído todos los fragmentos del contexto. El dato de 240 créditos suele estar presente en las memorias de verificación."
    "8. Utiliza siempre Markdown para estructurar tus respuestas. Usa negritas para términos importantes, listas con viñetas para enumeraciones y asegúrate de dejar un espacio de línea entre párrafos para facilitar la lectura."
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
        historial: list[Mapping[str, Any]],
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

        # Construimos los mensajes: sistema + historial previo + pregunta actual
        messages = [("system", SYSTEM_PROMPT)]

        for msg in historial:
            messages.append((msg.role if hasattr(msg, 'role') else msg["role"],
                            msg.content if hasattr(msg, 'content') else msg["content"]))

        messages.append(("human", f"Contexto relevante:\n\n{context}\n\nPregunta del usuario:\n{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self._llm | StrOutputParser()

        return chain.invoke({})


__all__ = ["ChatEngine", "SYSTEM_PROMPT"]

