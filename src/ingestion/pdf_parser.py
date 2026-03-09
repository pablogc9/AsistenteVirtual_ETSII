from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document as LCDocument

from src.ingestion.processor import DocumentProcessor


class PDFProcessor:
    """
    Procesa PDFs con pdfplumber y archivos de texto plano (.txt),
    transformando su contenido en chunks compatibles con el pipeline RAG.

    Mejoras respecto a PyPDFLoader:
    - Extracción de tablas: cada celda se recupera como texto estructurado,
      lo que permite indexar datos tabulares (créditos, notas de corte, etc.).
    - Mejor manejo de codificación: pdfplumber gestiona internamente la
      conversión de bytes a texto, eliminando el problema del mojibake.
    - Soporte para .txt: permite añadir documentos curados manualmente a
      data/raw/ sin necesidad de convertirlos a PDF.
    """

    def __init__(
        self,
        chunk_size:    int = 800,
        chunk_overlap: int = 80,
    ) -> None:
        self._processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ── Extracción de PDFs ────────────────────────────────────────────────────

    @staticmethod
    def _extract_page_text(page: "pdfplumber.page.Page") -> str:
        """
        Extrae texto de una página combinando el flujo de texto normal con
        las celdas de todas las tablas detectadas.

        Estrategia:
          1. Texto libre → pdfplumber.extract_text() preserva el orden de
             lectura natural y maneja encodings complejos.
          2. Tablas     → extract_tables() recupera cada celda de forma
             estructurada. Las filas se formatean como
             'col1 | col2 | col3' para que el modelo las entienda.
          3. Ambos se concatenan con un separador claro.
        """
        # ── Texto libre ───────────────────────────────────────────────────────
        body_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        # ── Tablas ────────────────────────────────────────────────────────────
        table_lines: list[str] = []
        for table in page.extract_tables():
            for row in table:
                if not row:
                    continue
                cells = [
                    str(cell).strip().replace("\n", " ") if cell else ""
                    for cell in row
                ]
                non_empty = [c for c in cells if c]
                if non_empty:
                    table_lines.append(" | ".join(non_empty))

        if table_lines:
            body_text = body_text + "\n\n" + "\n".join(table_lines)

        return body_text

    def _process_single_pdf(self, pdf_path: Path) -> List[LCDocument]:
        """Carga un PDF y devuelve los chunks con metadatos de página."""
        filename    = pdf_path.name
        source_path = str(pdf_path)
        all_chunks: List[LCDocument] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    page_text = self._extract_page_text(page)
                    if not page_text.strip():
                        continue

                    page_chunks = self._processor.process(
                        title=filename,
                        url=source_path,
                        paragraphs=[page_text],
                    )

                    for chunk in page_chunks:
                        meta = dict(chunk.metadata or {})
                        meta.setdefault("filename",    filename)
                        meta.setdefault("source_path", source_path)
                        meta.setdefault("page_number", page_number)
                        chunk.metadata = meta  # type: ignore[attr-defined]

                    all_chunks.extend(page_chunks)

        except Exception as exc:
            print(f"[PDFProcessor] Error al procesar {pdf_path.name}: {exc}")

        return all_chunks

    # ── Extracción de archivos de texto plano ─────────────────────────────────

    def _process_text_file(self, txt_path: Path) -> List[LCDocument]:
        """
        Procesa un archivo .txt indexando CADA línea no vacía como un chunk
        independiente.

        Por qué línea a línea y no el archivo completo:
          Los archivos de datos curados (datos_maestros.txt) contienen hechos
          muy concretos y cortos. Si se unen y trocean a 800 chars, varios
          hechos quedan mezclados en el mismo chunk y la similitud vectorial con
          queries cortas ("Créditos software?") se diluye.
          Procesar cada línea por separado garantiza que la búsqueda vectorial
          encuentre el hecho exacto, ya que el embedding del chunk contiene
          solo ese hecho y nada más.
        """
        filename    = txt_path.name
        source_path = str(txt_path)

        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"[PDFProcessor] Error al leer {txt_path.name}: {exc}")
            return []

        lines = [ln.strip() for ln in raw_text.splitlines() if len(ln.strip()) >= 30]
        if not lines:
            return []

        all_chunks: List[LCDocument] = []
        for line in lines:
            chunks = self._processor.process(
                title=filename,
                url=source_path,
                paragraphs=[line],
            )
            for chunk in chunks:
                meta = dict(chunk.metadata or {})
                meta.setdefault("filename",    filename)
                meta.setdefault("source_path", source_path)
                chunk.metadata = meta  # type: ignore[attr-defined]
            all_chunks.extend(chunks)

        return all_chunks

    # ── Procesado de carpeta completa ─────────────────────────────────────────

    @staticmethod
    def _load_pdf_sources(folder_path: Path) -> dict:
        """
        Lee pdf_sources.json si existe (generado por crawl_etsi.py).
        Contiene, por cada PDF, la URL de la página web que lo enlazaba,
        lo que permite enriquecer los metadatos de los chunks.
        """
        sources_file = folder_path / "pdf_sources.json"
        if sources_file.exists():
            try:
                return json.loads(sources_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def process_folder(self, folder: str | Path) -> List[LCDocument]:
        """
        Procesa todos los PDFs y archivos .txt de una carpeta y devuelve
        la lista unificada de chunks listos para indexar.

        Si existe pdf_sources.json (generado por el crawler), añade a cada
        chunk los metadatos de la página web de origen del PDF.
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise ValueError(f"La ruta {folder_path} no es una carpeta válida.")

        all_chunks: List[LCDocument] = []
        pdf_sources = self._load_pdf_sources(folder_path)

        pdfs = sorted(folder_path.glob("*.pdf"))
        txts = [t for t in sorted(folder_path.glob("*.txt")) if t.name != "pdf_sources.json"]

        print(f"[PDFProcessor] {len(pdfs)} PDF(s) y {len(txts)} TXT(s) encontrados en {folder_path.name}/")
        if pdf_sources:
            print(f"[PDFProcessor] pdf_sources.json encontrado — {len(pdf_sources)} entrada(s) de origen web")

        for pdf_path in pdfs:
            print(f"  → PDF: {pdf_path.name}")
            chunks = self._process_single_pdf(pdf_path)

            # Enriquecer con metadatos de origen web si están disponibles
            source_meta = pdf_sources.get(pdf_path.name, {})
            if source_meta:
                for chunk in chunks:
                    meta = dict(chunk.metadata or {})
                    meta.setdefault("parent_url",   source_meta.get("parent_url", ""))
                    meta.setdefault("parent_title", source_meta.get("parent_title", ""))
                    meta.setdefault("pdf_url",      source_meta.get("source_url", ""))
                    chunk.metadata = meta  # type: ignore[attr-defined]

            all_chunks.extend(chunks)

        for txt_path in txts:
            print(f"  → TXT: {txt_path.name}")
            all_chunks.extend(self._process_text_file(txt_path))

        return all_chunks
