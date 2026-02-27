from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument

from src.ingestion.processor import DocumentProcessor


class PDFProcessor:
    """
    Procesa PDFs con PyPDFLoader y los transforma en chunks compatibles 
    con el resto del pipeline (DocumentProcessor + metadatos)
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
    ) -> None:
        self._processor = DocumentProcessor(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )

    def _process_single_pdf(self, pdf_path: Path) -> List[LCDocument]:
        """Carga un PDF, procesa cada página y devuelve los chunks resultantes 
        con metadatos: nombre de archivo, ruta y número de página.
        """
        loader = PyPDFLoader(str(pdf_path))
        pages: List[LCDocument] = loader.load()

        all_chunks: List[LCDocument] = []
        filename = pdf_path.name
        source_path = str(pdf_path)

        for page_doc in pages:
            page_text = page_doc.page_content or ""
            if not page_text.strip():
                continue

            # Número de página que expone PyPDFLoader (suele empezar en 0)
            page_number = page_doc.metadata.get("page", None)

            # Usamos el DocumentoProcessor para hacer el chunking del texto de la página
            # titulo = nombre de archivo
            # url = ruta al archivo
            page_chunks = self._processor.process(
                title = filename,
                url = source_path,
                paragraphs = [page_text],
            )

            # Añadimos metadatos de PDF a cada chunk
            for chunk in page_chunks:
                meta = dict(chunk.metadata or {})
                meta.setdefault("filename", filename)
                meta.setdefault("source_path", source_path)
                if page_number is not None:
                    meta.setdefault("page_number", page_number)
                chunk.metadata = meta # type: ignore[attr-defined]

            all_chunks.extend(page_chunks)

        return all_chunks

    def process_folder(self, folder: str | Path) -> List[LCDocument]:
        """
        Procesa todos los PDFs de una carpeta y devuelve una lista
        con todos los chunks generados.
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise ValueError(f"La ruta {folder_path} no es una carpeta válida.")

        all_chunks: List[LCDocument] = []

        for pdf_path in sorted(folder_path.glob("*.pdf")):
            pdf_chunks = self._process_single_pdf(pdf_path)
            all_chunks.extend(pdf_chunks)

        return all_chunks