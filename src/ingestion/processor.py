from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Procesa el contenido extraído por el scraper:
    - Limpia párrafos irrelevantes.
    - Aplica chunking con RecursiveCharacterTextSplitter.
    - Mantiene metadatos de título y URL por chunk.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        min_paragraph_length: int = 100,
        menu_words: Iterable[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_paragraph_length = min_paragraph_length
        self.menu_words = [
            w.lower()
            for w in (
                menu_words
                if menu_words is not None
                else ["sesión", "iduma", "cerrar"]
            )
        ]

        # Inicializamos el splitter. Hace los cortes siguiendo un orden de prioridad: ["\n\n", "\n", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def clean_paragraphs(self, paragraphs: Iterable[str]) -> List[str]:
        """
        Elimina párrafos demasiado cortos o que contienen palabras típicas de menús.

        - Descarta textos con longitud < min_paragraph_length.
        - Descarta textos que contengan alguna de las palabras configuradas en menu_words
          (comparación insensible a mayúsculas/minúsculas).
        """
        cleaned: List[str] = []

        for p in paragraphs:
            if not p:
                continue

            texto = p.strip()
            if len(texto) < self.min_paragraph_length:
                continue

            lower = texto.lower()
            if any(word in lower for word in self.menu_words):
                continue

            cleaned.append(texto)

        return cleaned

    def _build_base_document(
        self,
        title: str,
        url: str,
        paragraphs: Iterable[str],
    ) -> Document:
        """
        Construye un único Document con todo el texto limpio y metadatos base.
        Este Document se usará como entrada al splitter para generar los chunks.
        """
        joined_text = "\n\n".join(paragraphs)

        return Document(
            page_content=joined_text,
            metadata={
                "title": title,
                "source": url,
            },
        )

    def process(
        self,
        title: str,
        url: str,
        paragraphs: Iterable[str],
    ) -> List[Document]:
        """
        Pipeline completo:
        1. Limpia los párrafos.
        2. Genera chunks con RecursiveCharacterTextSplitter.
        3. Devuelve una lista de Documents, cada uno con título y URL en metadata.
        """
        cleaned_paragraphs = self.clean_paragraphs(paragraphs)

        if not cleaned_paragraphs:
            return []

        base_doc = self._build_base_document(title, url, cleaned_paragraphs)
        chunks = self._splitter.split_documents([base_doc])

        return chunks


__all__ = ["DocumentProcessor"]