from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document as LCDocument

from src.ingestion.scraper import obtener_titulo_y_parrafos
from src.ingestion.processor import DocumentProcessor
from src.ingestion.pdf_parser import PDFProcessor
from src.database.vector_store import VectorStoreManager


def ingest_web_pages(processor: DocumentProcessor) -> List[LCDocument]:
    """
    Procesa varias URLs fijas de la ETSI y devuelve la lista de chunks.
    """
    urls = [
        "https://www.uma.es/etsi-informatica/",
        "https://www.uma.es/grado-en-ingenieria-del-software/",
        "https://www.uma.es/grado-en-ingenieria-informatica/",
    ]

    all_chunks: List[LCDocument] = []

    for url in urls:
        print(f"\n[WEB] Procesando URL: {url}")
        try:
            title, paragraphs = obtener_titulo_y_parrafos(url)
            chunks = processor.process(title, url, paragraphs)
            print(f"  - Chunks generados: {len(chunks)}")
            all_chunks.extend(chunks)
        except Exception as exc:  # pragma: no cover - logging simple de errores
            print(f"  ! Error procesando {url}: {exc}")

    return all_chunks


def ingest_pdfs(pdf_folder: str | Path, pdf_processor: PDFProcessor) -> List[LCDocument]:
    """
    Procesa todos los PDFs de una carpeta y devuelve la lista de chunks.
    """
    print(f"\n[PDF] Procesando carpeta: {pdf_folder}")

    try:
        chunks = pdf_processor.process_folder(pdf_folder)
    except Exception as exc:  # pragma: no cover - logging simple de errores
        print(f"  ! Error procesando PDFs en {pdf_folder}: {exc}")
        return []

    print(f"  - Chunks generados desde PDFs: {len(chunks)}")
    return chunks


def main() -> None:
    # Cargar variables de entorno (por si fueran necesarias para otros componentes)
    load_dotenv()

    # Instanciar procesadores y vector store
    text_processor = DocumentProcessor()
    pdf_processor = PDFProcessor()
    vector_store = VectorStoreManager()

    # 1) Ingesta de páginas web fijas
    web_chunks = ingest_web_pages(text_processor)

    # 2) Ingesta de PDFs en data/raw
    pdf_chunks = ingest_pdfs("data/raw", pdf_processor)

    # 3) Unificar y guardar todo en el mismo VectorStoreManager
    all_chunks = web_chunks + pdf_chunks
    print(f"\n[TOTAL] Chunks a almacenar en el vector store: {len(all_chunks)}")

    if not all_chunks:
        print("No hay chunks para almacenar. Saliendo.")
        return

    vector_store.add_documents(all_chunks)
    print("\n[OK] Ingesta combinada (web + PDFs) completada y almacenada en ChromaDB.")


if __name__ == "__main__":
    main()

