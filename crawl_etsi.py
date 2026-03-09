#!/usr/bin/env python3
"""
Crawler e Ingesta Híbrida — ETSI Informática (UMA)
===================================================

Recorre el dominio de la ETSI extrayendo:
  · Contenido HTML  → chunking + ingesta directa en ChromaDB
  · Archivos PDF    → descarga en data/raw/ con metadatos de origen

Características:
  · Estado persistente en data/crawl_state.json (reanudable si se interrumpe)
  · Deduplicación de URLs y PDFs (hash SHA-256 del contenido)
  · Rate-limiting configurable para no saturar el servidor
  · Filtrado de ruido HTML (menús, pies de página, scripts)
  · Registro de la página web que enlazaba cada PDF (vital para el contexto)

Uso:
    python crawl_etsi.py               # Crawl completo (reanuda si hay estado)
    python crawl_etsi.py --reset       # Empieza desde cero (borra estado previo)
    python crawl_etsi.py --dry-run     # Solo imprime URLs descubiertas, no ingesta
    python crawl_etsi.py --no-ingest   # Descarga todo pero no toca ChromaDB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


# ── Configuración ──────────────────────────────────────────────────────────────

# URLs de inicio del crawl
START_URLS: list[str] = [
    "https://www.uma.es/etsi-informatica/",
    "https://www.uma.es/informatica-matematicas/",
]

# Solo se siguen enlaces que empiezan por alguno de estos prefijos.
# Añadir aquí cualquier sección adicional de la web que interese indexar.
ALLOWED_PREFIXES: list[str] = [
    "https://www.uma.es/etsi-informatica/",
    "https://www.uma.es/etsi-informatica",                              # sin trailing slash
    "https://www.uma.es/grado-en-ingenieria-del-software/",
    "https://www.uma.es/grado-en-ingenieria-informatica/",
    "https://www.uma.es/grado-en-ciberseguridad-e-inteligencia-artificial/",
    "https://www.uma.es/informatica-matematicas/",
]

PDF_DIR    = Path("data/raw")
STATE_FILE = Path("data/crawl_state.json")
PDF_SOURCES = PDF_DIR / "pdf_sources.json"

DELAY_SECONDS = 1.5   # pausa entre peticiones (segundos)
MAX_PAGES     = 500   # límite de seguridad para evitar loops infinitos
TIMEOUT       = 12    # segundos de timeout por petición

HEADERS = {
    "User-Agent": (
        "AsistenteETSI-Crawler/1.0 "
        "(TFG UMA research; respectful crawling; contact: estudiante@uma.es)"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
}

# Etiquetas HTML que generan ruido (menús, pies, scripts…)
NOISE_TAGS = [
    "nav", "header", "footer", "aside",
    "script", "style", "noscript", "iframe", "form",
]

# Extensiones que se descartan aunque estén dentro del dominio
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".avi", ".zip", ".rar", ".exe",
    ".css", ".js", ".xml", ".json",
}


# ── Estado del crawl ──────────────────────────────────────────────────────────

class CrawlState:
    """
    Gestiona el estado persistente del crawler para permitir reanudar
    sesiones interrumpidas sin reprocesar URLs ya visitadas.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self.visited_urls:    set[str] = set()
        self.downloaded_pdfs: set[str] = set()  # SHA-256 del contenido
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self.visited_urls    = set(data.get("visited_urls", []))
                self.downloaded_pdfs = set(data.get("downloaded_pdfs", []))
                print(f"[Estado] Reanudando: {len(self.visited_urls)} URLs visitadas, "
                      f"{len(self.downloaded_pdfs)} PDFs descargados.")
            except Exception:
                pass

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({
                "visited_urls":    list(self.visited_urls),
                "downloaded_pdfs": list(self.downloaded_pdfs),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.visited_urls    = set()
        self.downloaded_pdfs = set()
        if self._path.exists():
            self._path.unlink()
        print("[Estado] Estado reseteado.")


# ── Utilidades ────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_allowed(url: str) -> bool:
    """Devuelve True si la URL pertenece al dominio permitido."""
    return any(url.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def is_skippable(url: str) -> bool:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext in SKIP_EXTENSIONS


def safe_filename(url: str) -> str:
    """Convierte una URL de PDF en un nombre de archivo seguro."""
    name = Path(urlparse(url).path).name
    return name if name else sha256_bytes(url.encode())[:16] + ".pdf"


def extract_links(soup: BeautifulSoup, current_url: str) -> set[str]:
    """Extrae todos los enlaces absolutos de la página."""
    links: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        absolute = urljoin(current_url, href).split("#")[0].split("?")[0]
        if absolute and not is_skippable(absolute):
            links.add(absolute)
    return links


def extract_content(soup: BeautifulSoup, url: str) -> tuple[str, list[str]]:
    """
    Limpia el HTML y extrae el título + lista de párrafos del contenido central.

    Estrategia:
      1. Elimina etiquetas de ruido (nav, footer, scripts…).
      2. Busca el contenedor principal: <main>, <article>, div#content, etc.
      3. Extrae texto de <p>, <li>, <h1-h4>, <td> del contenedor.
    """
    # Eliminar ruido
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    # Título
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url

    # Contenedor principal
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(id="main-content")
        or soup.find(class_="content")
        or soup.body
    )
    if main is None:
        return title, []

    paragraphs: list[str] = []
    for tag in main.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "th"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    return title, paragraphs


# ── PDF metadata store ────────────────────────────────────────────────────────

def load_pdf_sources() -> dict[str, dict]:
    if PDF_SOURCES.exists():
        try:
            return json.loads(PDF_SOURCES.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_pdf_sources(data: dict[str, dict]) -> None:
    PDF_SOURCES.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Crawler principal ─────────────────────────────────────────────────────────

class ETSICrawler:

    def __init__(
        self,
        state:     CrawlState,
        ingest:    bool = True,
        dry_run:   bool = False,
    ) -> None:
        self._state   = state
        self._ingest  = ingest
        self._dry_run = dry_run

        if ingest and not dry_run:
            from src.ingestion.processor import DocumentProcessor
            from src.database.vector_store import VectorStoreManager
            self._processor    = DocumentProcessor()
            self._vector_store = VectorStoreManager()
        else:
            self._processor    = None
            self._vector_store = None

        PDF_DIR.mkdir(parents=True, exist_ok=True)
        self._pdf_sources = load_pdf_sources()

        self._session = requests.Session()
        self._session.headers.update(HEADERS)

        self._html_chunks  = 0
        self._pdfs_new     = 0
        self._pdfs_skip    = 0
        self._pages_ok     = 0
        self._pages_error  = 0

    # ── Peticiones HTTP ───────────────────────────────────────────────────────

    def _get(self, url: str) -> requests.Response | None:
        try:
            response = self._session.get(url, timeout=TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            print(f"  [ERROR] {url}: {exc}")
            self._pages_error += 1
            return None

    # ── Procesado de PDF ──────────────────────────────────────────────────────

    def _handle_pdf(
        self,
        url:          str,
        parent_url:   str,
        parent_title: str,
    ) -> None:
        """Descarga el PDF si no se ha descargado ya (por hash de contenido)."""
        if self._dry_run:
            print(f"  [PDF-dry] {url}")
            return

        response = self._get(url)
        if response is None:
            return

        content_hash = sha256_bytes(response.content)
        if content_hash in self._state.downloaded_pdfs:
            self._pdfs_skip += 1
            print(f"  [PDF-dup] {url}")
            return

        filename = safe_filename(url)
        dest = PDF_DIR / filename

        # Si el nombre ya existe pero el hash es diferente, añadir sufijo
        if dest.exists():
            stem   = dest.stem
            suffix = dest.suffix
            dest   = PDF_DIR / f"{stem}_{content_hash[:8]}{suffix}"

        dest.write_bytes(response.content)
        self._state.downloaded_pdfs.add(content_hash)

        # Guardar metadatos de origen
        self._pdf_sources[filename] = {
            "source_url":    url,
            "parent_url":    parent_url,
            "parent_title":  parent_title,
        }
        save_pdf_sources(self._pdf_sources)

        self._pdfs_new += 1
        print(f"  [PDF] Descargado → {filename} (desde: {parent_url})")

    # ── Procesado de página HTML ──────────────────────────────────────────────

    def _handle_html(self, url: str, response: requests.Response) -> set[str]:
        """
        Extrae contenido de la página, lo ingesta en ChromaDB (si procede)
        y devuelve el conjunto de enlaces descubiertos.
        """
        try:
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            print(f"  [ERROR parse] {url}: {exc}")
            return set()

        title, paragraphs = extract_content(soup, url)
        links = extract_links(soup, url)

        if self._dry_run:
            print(f"  [HTML-dry] {url} | {len(paragraphs)} párrafos | {len(links)} enlaces")
            return links

        if paragraphs and self._ingest and self._processor and self._vector_store:
            chunks = self._processor.process(title=title, url=url, paragraphs=paragraphs)
            if chunks:
                self._vector_store.add_documents(chunks)
                self._html_chunks += len(chunks)
                print(f"  [HTML] {url} → {len(chunks)} chunks")
            else:
                print(f"  [HTML] {url} → sin contenido útil")
        else:
            print(f"  [HTML] {url} → {len(paragraphs)} párrafos (sin ingesta)")

        return links

    # ── Bucle principal ───────────────────────────────────────────────────────

    def run(self) -> None:
        queue: deque[tuple[str, str, str]] = deque()  # (url, parent_url, parent_title)

        for start_url in START_URLS:
            if start_url not in self._state.visited_urls:
                queue.append((start_url, "", ""))

        pages_processed = 0

        print(f"\n{'='*60}")
        print(f"  Iniciando crawl — {len(queue)} URL(s) de inicio")
        print(f"  Delay: {DELAY_SECONDS}s | Límite: {MAX_PAGES} páginas")
        print(f"  Modo: {'dry-run' if self._dry_run else ('con ingesta' if self._ingest else 'sin ingesta')}")
        print(f"{'='*60}\n")

        while queue and pages_processed < MAX_PAGES:
            url, parent_url, parent_title = queue.popleft()

            if url in self._state.visited_urls:
                continue

            self._state.visited_urls.add(url)
            pages_processed += 1

            print(f"[{pages_processed}/{MAX_PAGES}] {url}")

            # ── PDF ──────────────────────────────────────────────────────────
            if is_pdf_url(url):
                self._handle_pdf(url, parent_url, parent_title)
                time.sleep(DELAY_SECONDS)
                continue

            # ── HTML ─────────────────────────────────────────────────────────
            response = self._get(url)
            if response is None:
                time.sleep(DELAY_SECONDS)
                continue

            # Comprobar que es HTML (podría ser un PDF sin extensión)
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" in content_type:
                self._handle_pdf(url, parent_url, parent_title)
                time.sleep(DELAY_SECONDS)
                continue

            if "text/html" not in content_type:
                print(f"  [SKIP] Content-Type no soportado: {content_type}")
                time.sleep(DELAY_SECONDS)
                continue

            # Obtener título antes de parsear (para metadatos de PDFs hijos)
            try:
                soup_tmp = BeautifulSoup(response.text, "html.parser")
                page_title = soup_tmp.title.string.strip() if soup_tmp.title and soup_tmp.title.string else url
            except Exception:
                page_title = url

            new_links = self._handle_html(url, response)
            self._pages_ok += 1

            # Encolar enlaces nuevos
            for link in new_links:
                if link not in self._state.visited_urls and is_allowed(link):
                    queue.append((link, url, page_title))

            # Guardar estado periódicamente (cada 10 páginas)
            if pages_processed % 10 == 0:
                self._state.save()

            time.sleep(DELAY_SECONDS)

        # Guardar estado final
        self._state.save()

        print(f"\n{'='*60}")
        print("  CRAWL COMPLETADO")
        print(f"  Páginas OK:       {self._pages_ok}")
        print(f"  Páginas con error:{self._pages_error}")
        print(f"  PDFs nuevos:      {self._pdfs_new}")
        print(f"  PDFs duplicados:  {self._pdfs_skip}")
        print(f"  Chunks HTML:      {self._html_chunks}")
        print(f"{'='*60}\n")

        if self._pdfs_new > 0:
            print(f"  → {self._pdfs_new} PDF(s) descargados en {PDF_DIR}/")
            print("  → Ejecuta 'python main_ingestion.py' para indexarlos en ChromaDB.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawler e ingesta híbrida de la ETSI Informática (UMA)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra el estado guardado y empieza el crawl desde cero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Descubre URLs y PDFs pero no descarga ni modifica ChromaDB.",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Descarga PDFs y procesa HTML pero no añade nada a ChromaDB.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_SECONDS,
        help=f"Segundos de pausa entre peticiones (default: {DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES,
        help=f"Número máximo de páginas a procesar (default: {MAX_PAGES}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Aplicar argumentos a las constantes globales
    globals()["DELAY_SECONDS"] = args.delay
    globals()["MAX_PAGES"]     = args.max_pages

    state = CrawlState(STATE_FILE)
    if args.reset:
        state.reset()

    crawler = ETSICrawler(
        state=state,
        ingest=not args.no_ingest,
        dry_run=args.dry_run,
    )
    crawler.run()
