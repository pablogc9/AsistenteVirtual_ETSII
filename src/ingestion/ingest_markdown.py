"""
ingest_markdown.py
==================
Pipeline de ingesta definitivo para la colección ChromaDB 'etsi_hibrida'.

Lee los archivos .md de data/processed/ (generados por pdf_to_markdown.py)
y los ingesta en ChromaDB con chunking jerárquico en dos pasos:

  Paso 1 – MarkdownHeaderTextSplitter
    Divide cada documento siguiendo la jerarquía de encabezados (#, ##, ###).
    Cada sección resultante hereda automáticamente la ruta de headings completa
    en sus metadatos (Header 1 / Header 2 / Header 3).

  Paso 2 – RecursiveCharacterTextSplitter
    Si una sección supera MAX_SECTION_CHARS, la subdivide preservando el overlap
    para evitar cortar párrafos o filas de tabla a la mitad.

Metadatos de cada chunk:
    title        – título del documento (primer H1 o nombre del fichero)
    section      – ruta de encabezados "H1 > H2 > H3"
    source_file  – nombre del .md de origen
    source_url   – URL original de descarga (de pdf_sources.json si existe)
    chunk_index  – posición del chunk dentro de la sección

Uso:
    python -m src.ingestion.ingest_markdown            # ingesta completa
    python -m src.ingestion.ingest_markdown --reset    # borra colección y re-ingesta
    python -m src.ingestion.ingest_markdown --dry-run  # muestra estadísticas sin subir
    python -m src.ingestion.ingest_markdown --file data/processed/Reglamento_TFG.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Proyecto ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.database.vector_store import VectorStoreManager  # noqa: E402

# ── Constantes configurables ──────────────────────────────────────────────────

PROCESSED_DIR    = _ROOT / "data" / "processed"
PDF_SOURCES_JSON = _ROOT / "data" / "raw" / "pdf_sources.json"
WEB_SOURCES_JSON = _ROOT / "data" / "raw" / "web_sources.json"

COLLECTION_NAME  = "etsi_hibrida"

# Tamaño máximo de una sección antes del segundo corte
MAX_SECTION_CHARS = 1_500
# Overlap del segundo corte (RecursiveCharacterTextSplitter)
CHUNK_OVERLAP     = 150
# Longitud mínima para indexar un chunk (filtrar ruido residual)
MIN_CHUNK_CHARS   = 60

# Encabezados que MarkdownHeaderTextSplitter debe usar como pivote de corte
HEADERS_TO_SPLIT = [
    ("#",   "Header 1"),
    ("##",  "Header 2"),
    ("###", "Header 3"),
]

# ── Separadores jerárquicos del segundo splitter ──────────────────────────────
# Orden: tabla Markdown → párrafo → oración → espacio → carácter
_RECURSIVE_SEPARATORS = [
    "\n\n",   # párrafo / bloque
    "\n|",    # inicio de fila de tabla Markdown
    "\n",     # línea
    ". ",     # fin de oración
    " ",
    "",
]

# ── Carga del mapa de URLs ─────────────────────────────────────────────────────

def _load_url_map() -> dict[str, str]:
    """
    Devuelve un dict de URLs de origen combinando dos fuentes:
      · pdf_sources.json → claves por nombre de PDF  ({nombre.pdf: url})
      · web_sources.json → claves por nombre de .md   ({web__x.md: url})
    """
    url_map: dict[str, str] = {}

    if PDF_SOURCES_JSON.exists():
        try:
            raw = json.loads(PDF_SOURCES_JSON.read_text(encoding="utf-8"))
            url_map.update(
                {k: v.get("source_url", "") for k, v in raw.items() if isinstance(v, dict)}
            )
        except Exception:
            pass

    if WEB_SOURCES_JSON.exists():
        try:
            raw = json.loads(WEB_SOURCES_JSON.read_text(encoding="utf-8"))
            url_map.update({k: v for k, v in raw.items() if isinstance(v, str)})
        except Exception:
            pass

    return url_map


# ── Extracción del título del documento ───────────────────────────────────────

def _extract_title(md_text: str, filename_stem: str) -> str:
    """
    Busca el primer encabezado H1 en el Markdown.
    Si no existe, usa el nombre del fichero (URL-decoded y limpio).
    """
    for line in md_text.splitlines():
        m = re.match(r"^#\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    # Fallback: decodificar nombre de fichero (%20 → espacio, etc.)
    from urllib.parse import unquote
    return unquote(filename_stem).replace("_", " ").replace("-", " ").strip()


# ── Construcción de la ruta de sección ────────────────────────────────────────

def _build_section(metadata: dict[str, Any]) -> str:
    """
    Combina los encabezados jerárquicos en una ruta legible:
      "Capítulo I > Artículo 3 > Apartado b"
    """
    parts = []
    for key in ("Header 1", "Header 2", "Header 3"):
        val = metadata.get(key, "").strip()
        if val:
            parts.append(val)
    return " > ".join(parts) if parts else ""


# ── Pipeline de chunking ───────────────────────────────────────────────────────

def _split_markdown(md_text: str, title: str, source_file: str, source_url: str) -> list[Document]:
    """
    Aplica los dos pasos de chunking y enriquece los metadatos.

    Returns:
        Lista de Documents listos para indexar.
    """
    # ── Paso 1: corte jerárquico por encabezados ───────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,   # conservar el texto del encabezado en el chunk
        return_each_line=False,
    )
    sections: list[Document] = header_splitter.split_text(md_text)

    # ── Paso 2: sub-corte de secciones largas ─────────────────────────────────
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SECTION_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_RECURSIVE_SEPARATORS,
    )

    final_chunks: list[Document] = []
    for section in sections:
        content = section.page_content.strip()
        if not content:
            continue

        section_path = _build_section(section.metadata)

        if len(content) > MAX_SECTION_CHARS:
            sub_chunks = char_splitter.split_text(content)
        else:
            sub_chunks = [content]

        for idx, chunk_text in enumerate(sub_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            final_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "title":       title,
                        "section":     section_path,
                        "source_file": source_file,
                        "source_url":  source_url,
                        "chunk_index": idx,
                    },
                )
            )

    return final_chunks


# ── Ingesta de un único fichero .md ───────────────────────────────────────────

def process_md_file(
    md_path:  Path,
    url_map:  dict[str, str],
) -> list[Document]:
    """Lee un .md y devuelve sus chunks con metadatos completos."""
    md_text = md_path.read_text(encoding="utf-8")
    title   = _extract_title(md_text, md_path.stem)

    # Resolver la URL de origen:
    #   · PDFs: el .md comparte stem con el .pdf  → buscar "<stem>.pdf"
    #   · Webs: el .md (web__*.md) se registra con su propio nombre en web_sources
    source_url = url_map.get(md_path.stem + ".pdf", "") or url_map.get(md_path.name, "")

    return _split_markdown(
        md_text    = md_text,
        title      = title,
        source_file= md_path.name,
        source_url = source_url,
    )


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run(
    md_files:   list[Path],
    reset:      bool,
    dry_run:    bool,
    collection: str,
    batch_size: int = 200,
) -> None:

    url_map = _load_url_map()

    print(f"\n{'='*65}")
    print(f"  Ingesta Markdown -> ChromaDB")
    print(f"  Coleccion : {collection}")
    print(f"  Archivos  : {len(md_files)}")
    print(f"  Modo      : {'DRY-RUN' if dry_run else 'RESET+INGEST' if reset else 'INCREMENTAL'}")
    print(f"{'='*65}\n")

    # ── Procesar ficheros → chunks ─────────────────────────────────────────────
    all_chunks: list[Document] = []
    for i, md_path in enumerate(sorted(md_files), 1):
        try:
            chunks = process_md_file(md_path, url_map)
            all_chunks.extend(chunks)
            print(f"  [{i:>3}/{len(md_files)}]  {md_path.name[:55]:<55}  {len(chunks):>4} chunks")
        except Exception as exc:
            print(f"  [{i:>3}/{len(md_files)}]  {md_path.name[:55]:<55}  ERROR: {exc}")

    # ── Estadísticas ───────────────────────────────────────────────────────────
    if all_chunks:
        avg_len = sum(len(c.page_content) for c in all_chunks) // len(all_chunks)
        max_len = max(len(c.page_content) for c in all_chunks)
        min_len = min(len(c.page_content) for c in all_chunks)
        print(f"\n  Total chunks : {len(all_chunks)}")
        print(f"  Long. media  : {avg_len} chars")
        print(f"  Min / Max    : {min_len} / {max_len} chars")

    if dry_run or not all_chunks:
        print("\n  [DRY-RUN] No se subio nada a ChromaDB.\n")
        return

    # ── Subida a ChromaDB ──────────────────────────────────────────────────────
    vsm = VectorStoreManager(collection_name=collection)

    if reset:
        print(f"\n  Borrando coleccion '{collection}'...")
        try:
            vsm._client.delete_collection(collection)
            # Recrear colección tras borrar
            vsm._collection = vsm._client.get_or_create_collection(
                name=collection,
                embedding_function=vsm._embedding_function,
            )
            print("  Coleccion recreada.")
        except Exception as exc:
            print(f"  [WARN] No se pudo borrar la coleccion: {exc}")

    print(f"\n  Subiendo {len(all_chunks)} chunks en lotes de {batch_size}...")
    uploaded = 0
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        vsm.add_documents(batch)
        uploaded += len(batch)
        pct = uploaded / len(all_chunks) * 100
        print(f"    {uploaded:>5}/{len(all_chunks)}  ({pct:.0f}%)")

    total_in_db = vsm._collection.count()
    print(f"\n  Documentos totales en '{collection}': {total_in_db}")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta de Markdown estructurado en ChromaDB."
    )
    parser.add_argument(
        "--file", "-f", type=Path, default=None,
        help="Ingestar un solo .md en lugar de toda la carpeta",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
        help=f"Carpeta con los .md (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--collection", "-c", default=COLLECTION_NAME,
        help=f"Nombre de la coleccion ChromaDB (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Borrar la coleccion antes de ingestar (re-ingesta limpia)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar estadisticas sin subir a ChromaDB",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Chunks por lote al subir a ChromaDB (default: 200)",
    )
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            print(f"[ERROR] No existe: {args.file}")
            sys.exit(1)
        md_files = [args.file]
    else:
        md_files = [
            p for p in args.processed_dir.glob("*.md")
            if not p.name.startswith(".")
        ]
        if not md_files:
            print(f"[ERROR] No se encontraron .md en {args.processed_dir}")
            print("        Ejecuta primero: python -m scripts.ingestion.pdf_to_markdown")
            sys.exit(1)

    run(
        md_files   = md_files,
        reset      = args.reset,
        dry_run    = args.dry_run,
        collection = args.collection,
        batch_size = args.batch_size,
    )


if __name__ == "__main__":
    main()
