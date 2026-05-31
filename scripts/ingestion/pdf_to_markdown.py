#!/usr/bin/env python3
"""
pdf_to_markdown.py
==================
Convertir PDFs de data/raw/ a Markdown en data/processed/.

Incluye limpieza de ruido institucional, procesado incremental por hash
y manifiesto JSON de conversiones.

Uso:
    python -m scripts.ingestion.pdf_to_markdown
    python -m scripts.ingestion.pdf_to_markdown --force
    python -m scripts.ingestion.pdf_to_markdown --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Forzar UTF-8 en consola Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import pdfplumber
except ImportError:
    print("[ERROR] pdfplumber no instalado. Ejecuta: pip install pdfplumber")
    sys.exit(1)

# ── Rutas ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/processed/.manifest.json")

# ── Patrones de ruido específicos de documentos UMA/ETSI ──────────────────────
# Cada patrón elimina una línea completa si hace match (MULTILINE)

_NOISE_PATTERNS: list[re.Pattern] = [
    # Números de página solos: "- 1 -", "– 2 –", "1", "Página 3 de 10"
    re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.MULTILINE),
    re.compile(r"^\s*[Pp][áa]gina\s+\d+\s*(de\s+\d+)?\s*$", re.MULTILINE),
    re.compile(r"^\s*[Pp]age\s+\d+\s*(of\s+\d+)?\s*$", re.MULTILINE),

    # Cabeceras institucionales repetidas
    re.compile(r"^\s*Universidad de M[aá]laga\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Escuela T[eé]cnica Superior de Ingenier[ií]a Inform[aá]tica\s*$",
               re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*ETSI\s*Inform[aá]tica\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*E\.T\.S\.I\.\s*Inform[aá]tica\s*$", re.MULTILINE | re.IGNORECASE),

    # Pies de página con dirección postal UMA
    re.compile(r"^\s*C/?\s*[Ll]ouis\s*[Pp]asteur.*$", re.MULTILINE),
    re.compile(r"^\s*Campus\s+de\s+[Tt]eatinos.*$", re.MULTILINE),
    re.compile(r"^\s*29071\s+M[aá]laga.*$", re.MULTILINE),

    # URLs sueltas en encabezados/pies
    re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE),
    re.compile(r"^\s*www\.\S+\s*$", re.MULTILINE),

    # Fechas de encabezado tipo "Málaga, 15 de enero de 2024"
    re.compile(
        r"^\s*M[aá]laga,?\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),

    # Líneas de solo guiones/puntos (separadores visuales ya representados en MD)
    re.compile(r"^\s*[-_=·•]{4,}\s*$", re.MULTILINE),
]

# Bloques repetidos más de N veces seguidos (cabeceras que se repiten cada página)
_MAX_CONSECUTIVE_REPEATS = 2


# ── Limpieza ───────────────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """
    Aplicar limpieza de ruido al Markdown extraído con pdfplumber.

    Pasos:
      1. Elimina líneas de ruido institucional/navegación.
      2. Colapsa bloques de líneas en blanco excesivos (> 2 seguidas).
      3. Elimina repetición de líneas idénticas consecutivas (cabeceras de página).
    """
    # ── 1. Eliminar líneas de ruido ────────────────────────────────────────────
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # ── 2. Colapsar líneas en blanco excesivas ─────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── 3. Eliminar líneas idénticas repetidas consecutivamente ───────────────
    # (encabezados que pymupdf4llm detecta en cada página)
    lines     = text.splitlines()
    cleaned   = []
    prev_line = None
    streak    = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            streak += 1
            if streak < _MAX_CONSECUTIVE_REPEATS:
                cleaned.append(line)
        else:
            streak    = 0
            prev_line = stripped
            cleaned.append(line)
    text = "\n".join(cleaned)

    # ── 4. Limpieza final de espacios al inicio/final ──────────────────────────
    return text.strip() + "\n"


# ── Conversión de un PDF ───────────────────────────────────────────────────────

def _table_to_markdown(table: list[list]) -> str:
    """Convierte una tabla de pdfplumber (lista de listas) a formato Markdown pipe."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").replace("\n", " ").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "|".join(" --- " for _ in cells) + "|")
    return "\n".join(rows)


def _is_heading(line: str, char_sizes: list[float], median_size: float) -> str | None:
    """
    Heurística de detección de headings basada en tamaño de fuente y formato.
    Devuelve '#', '##', '###' o None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return None
    if not char_sizes:
        return None
    avg_size = sum(char_sizes) / len(char_sizes)
    ratio    = avg_size / max(median_size, 1)
    # Líneas en MAYÚSCULAS cortas también son candidatos a heading
    all_caps = stripped.isupper() and len(stripped.split()) <= 10
    if ratio >= 1.6 or (all_caps and ratio >= 1.2):
        return "#"
    if ratio >= 1.3:
        return "##"
    if ratio >= 1.15 or (all_caps and ratio >= 1.0 and len(stripped) < 80):
        return "###"
    return None


def _extract_page_markdown(page) -> str:
    """
    Extrae el contenido de una página pdfplumber como Markdown.

    Estrategia:
      1. Detecta las bounding boxes de las tablas para excluir ese texto del flujo.
      2. Convierte tablas a Markdown pipe.
      3. Convierte el texto restante detectando headings por tamaño de fuente.
    """
    # ── Obtener bboxes de tablas para enmascarar su región ─────────────────────
    tables     = page.extract_tables()
    table_bboxes = [t.bbox for t in page.find_tables()] if hasattr(page, "find_tables") else []

    md_parts: list[str] = []

    # ── Texto con metadatos de fuente ──────────────────────────────────────────
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return ""

    # Calcular tamaño de fuente mediano para calibrar heurística de headings
    sizes = [w.get("size", 10) for w in words if w.get("size")]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

    # Reagrupar palabras en líneas (mismo top ± 2px)
    lines_dict: dict[float, list[dict]] = {}
    for w in words:
        # Omitir palabras dentro de una bbox de tabla
        wx, wy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        in_table = any(
            bx0 <= wx <= bx1 and by0 <= wy <= by1
            for bx0, by0, bx1, by1 in table_bboxes
        )
        if in_table:
            continue
        top_key = round(w["top"] / 2) * 2
        lines_dict.setdefault(top_key, []).append(w)

    prev_bottom = 0.0
    for top_key in sorted(lines_dict):
        line_words = sorted(lines_dict[top_key], key=lambda w: w["x0"])
        text       = " ".join(w["text"] for w in line_words).strip()
        char_sizes = [w.get("size", median_size) for w in line_words]

        # Insertar línea en blanco si hay salto vertical grande (> 1.5× altura típica)
        if prev_bottom and (top_key - prev_bottom) > median_size * 1.5:
            md_parts.append("")

        heading = _is_heading(text, char_sizes, median_size)
        if heading:
            md_parts.append(f"\n{heading} {text}\n")
        else:
            # Detectar ítems de lista por sangría o prefijo
            stripped = text.strip()
            if re.match(r"^[•·▪▸\-]\s", stripped):
                md_parts.append(f"- {stripped[2:].strip()}")
            elif re.match(r"^\w\)\s", stripped) or re.match(r"^\d+[.)]\s", stripped):
                md_parts.append(f"- {stripped}")
            else:
                md_parts.append(text)

        prev_bottom = line_words[-1]["bottom"] if line_words else top_key

    # ── Añadir tablas al final del texto de la página ──────────────────────────
    for table in tables:
        if table:
            md_parts.append("\n" + _table_to_markdown(table) + "\n")

    return "\n".join(md_parts)


def _convert_pdf(pdf_path: Path) -> tuple[str, dict]:
    """
    Convierte un PDF a Markdown limpio usando pdfplumber.

    Detecta tablas (→ Markdown pipe), headings (por tamaño de fuente),
    listas y texto corrido. Aplica limpieza de ruido institucional.
    """
    t0 = time.time()
    raw_parts: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_md = _extract_page_markdown(page)
            if page_md.strip():
                raw_parts.append(page_md)

    raw_md   = "\n\n---\n\n".join(raw_parts)
    clean_md = _clean_markdown(raw_md)
    elapsed  = round(time.time() - t0, 2)

    stats = {
        "raw_chars":        len(raw_md),
        "clean_chars":      len(clean_md),
        "noise_removed_pct": round((1 - len(clean_md) / max(len(raw_md), 1)) * 100, 1),
        "elapsed_s":        elapsed,
    }
    return clean_md, stats


# ── Manifiesto (seguimiento incremental) ──────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _already_processed(pdf_path: Path, manifest: dict, force: bool) -> bool:
    """Devuelve True si el PDF ya fue convertido y el contenido no ha cambiado."""
    if force:
        return False
    entry = manifest.get(pdf_path.name)
    if not entry:
        return False
    out_path = PROCESSED_DIR / (pdf_path.stem + ".md")
    if not out_path.exists():
        return False
    return entry.get("sha256") == _sha256(pdf_path)


# ── Pipeline principal ─────────────────────────────────────────────────────────

def run(
    pdf_paths:   list[Path],
    force:       bool,
    dry_run:     bool,
    out_dir:     Path = PROCESSED_DIR,
    manifest_p:  Path = MANIFEST_PATH,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_p)

    total      = len(pdf_paths)
    skipped    = 0
    converted  = 0
    errors     = 0

    print(f"\n{'='*65}")
    print(f"  PDF -> Markdown Pipeline  |  {total} archivo(s)")
    print(f"  Destino: {PROCESSED_DIR}")
    print(f"  Modo: {'DRY-RUN' if dry_run else 'FORCE' if force else 'INCREMENTAL'}")
    print(f"{'='*65}\n")

    for i, pdf_path in enumerate(sorted(pdf_paths), 1):
        prefix = f"  [{i:>3}/{total}]  {pdf_path.name[:55]:<55}"

        if _already_processed(pdf_path, manifest, force):
            print(f"{prefix}  OMITIDO (sin cambios)")
            skipped += 1
            continue

        if dry_run:
            print(f"{prefix}  PENDIENTE")
            continue

        try:
            md_text, stats = _convert_pdf(pdf_path)
        except Exception as exc:
            print(f"{prefix}  ERROR: {exc}")
            errors += 1
            continue

        out_path = out_dir / (pdf_path.stem + ".md")
        out_path.write_text(md_text, encoding="utf-8")

        # Actualizar manifiesto
        manifest[pdf_path.name] = {
            "sha256":          _sha256(pdf_path),
            "output":          str(out_path),
            "processed_at":    datetime.now().isoformat(),
            "raw_chars":       stats["raw_chars"],
            "clean_chars":     stats["clean_chars"],
            "noise_removed_pct": stats["noise_removed_pct"],
            "elapsed_s":       stats["elapsed_s"],
        }
        _save_manifest(manifest, manifest_p)

        noise_tag = f"  -{stats['noise_removed_pct']:.0f}% ruido"
        print(f"{prefix}  OK  ({stats['elapsed_s']}s){noise_tag}")
        converted += 1

    print(f"\n{'='*65}")
    print(f"  Convertidos : {converted}")
    print(f"  Omitidos    : {skipped}  (ya procesados, hash identico)")
    print(f"  Errores     : {errors}")
    if not dry_run and converted > 0:
        print(f"  Manifiesto  : {manifest_p}")
    print(f"{'='*65}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convierte PDFs a Markdown estructurado para RAG."
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="PDF concreto a procesar (por defecto: todos en data/raw/)",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Carpeta de entrada con los PDFs (default: {RAW_DIR})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Carpeta de salida para los .md (default: {PROCESSED_DIR})",
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-procesar aunque el .md ya exista",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra qué se procesaria sin generar archivos",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out_dir    = args.out_dir
    manifest_p = out_dir / ".manifest.json"

    if args.input:
        if not args.input.exists():
            print(f"[ERROR] Archivo no encontrado: {args.input}")
            sys.exit(1)
        pdf_paths = [args.input]
    else:
        pdf_paths = list(args.raw_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"[ERROR] No se encontraron PDFs en {args.raw_dir}")
            sys.exit(1)

    run(pdf_paths, force=args.force, dry_run=args.dry_run,
        out_dir=out_dir, manifest_p=manifest_p)
