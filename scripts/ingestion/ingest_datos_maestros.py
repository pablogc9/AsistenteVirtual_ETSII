#!/usr/bin/env python3
"""
ingest_datos_maestros.py
========================
Ingesta el fichero de datos curados `data/raw/datos_maestros.txt` en la
colección ChromaDB «etsi_hibrida», con granularidad de **un chunk por línea**.

Cada línea del fichero es un hecho autocontenido (p. ej. "El Grado en Ingeniería
del Software tiene 240 créditos ECTS..."). Indexarlas por separado maximiza la
precisión de recuperación para preguntas concretas, ya que cada hecho se embebe
de forma independiente sin diluirse en un bloque mayor.

Es idempotente: los IDs de ChromaDB se derivan del hash del contenido, así que
re-ejecutarlo no genera duplicados.

Uso (desde la raíz del proyecto):
    python -m scripts.ingestion.ingest_datos_maestros
    python -m scripts.ingestion.ingest_datos_maestros --file otro_fichero.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from langchain_core.documents import Document  # noqa: E402

from src.database.vector_store import VectorStoreManager  # noqa: E402

DEFAULT_SOURCE = _ROOT / "data" / "raw" / "datos_maestros.txt"
COLLECTION     = "etsi_hibrida"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta datos_maestros.txt en etsi_hibrida (un chunk por línea)."
    )
    parser.add_argument(
        "--file", "-f", type=Path, default=DEFAULT_SOURCE,
        help=f"Fichero de hechos curados (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--collection", "-c", default=COLLECTION,
        help=f"Colección destino (default: {COLLECTION})",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"[ERROR] No existe el fichero: {args.file}")
        sys.exit(1)

    lines = [
        line.strip()
        for line in args.file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        print(f"[AVISO] {args.file} no tiene líneas con contenido. Nada que ingestar.")
        return

    docs = [
        Document(
            page_content=line,
            metadata={
                "title":       "Datos maestros ETSI Informática",
                "section":     "",
                "source_file": args.file.name,
                "source_url":  "",
                "chunk_index": idx,
            },
        )
        for idx, line in enumerate(lines)
    ]

    vsm = VectorStoreManager(collection_name=args.collection)
    vsm.add_documents(docs)

    print(f"[OK] {len(docs)} hechos curados ingestados en '{args.collection}'.")
    print(f"     Origen: {args.file}")


if __name__ == "__main__":
    main()
