#!/usr/bin/env python3
"""
run_baseline_eval.py
====================
Evalúa el sistema RAG contra el dataset de ground truth.

Flujo por pregunta:
  1. POST /ask  → captura answer, sources, log_id
  2. Calcula métricas heurísticas simples (sin LLM externo)
  3. Guardar todo en data/eval/results_hybrid_v2.json

Uso (ejecutar desde la raíz del proyecto):
    python -m scripts.evaluation.run_baseline_eval                     # todas
    python -m scripts.evaluation.run_baseline_eval --category datos_maestros
    python -m scripts.evaluation.run_baseline_eval --delay 2           # delay peticiones
    python -m scripts.evaluation.run_baseline_eval --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Configuración por defecto ──────────────────────────────────────────────────

DEFAULT_API   = "http://127.0.0.1:8000"
DATASET_PATH  = Path("data/eval/eval_dataset.json")
RESULTS_PATH = Path("data/eval/results_hybrid_v2.json")
NO_CONTEXT_SIGNAL = "no he encontrado información"   # fragmento de NO_CONTEXT_MSG


# ── Métricas heurísticas ───────────────────────────────────────────────────────

def _no_context_response(answer: str) -> bool:
    """True si el bot devolvió el mensaje de 'sin contexto'."""
    return NO_CONTEXT_SIGNAL.lower() in answer.lower()


def _keyword_overlap(answer: str, ground_truth: str) -> float:
    """
    Cobertura de palabras clave: fracción de tokens importantes del ground_truth
    que aparecen en la respuesta del bot.

    Solo considera tokens de ≥4 caracteres para ignorar artículos y preposiciones.
    Rango: 0.0 – 1.0
    """
    gt_tokens = {t.lower() for t in ground_truth.split() if len(t) >= 4}
    if not gt_tokens:
        return 0.0
    ans_lower = answer.lower()
    hits = sum(1 for t in gt_tokens if t in ans_lower)
    return round(hits / len(gt_tokens), 3)


def _answer_length(answer: str) -> int:
    """Número de palabras en la respuesta."""
    return len(answer.split())


# ── Llamada a la API ───────────────────────────────────────────────────────────

def call_ask(api_base: str, question: str, timeout: int = 60) -> dict:
    """
    Llama al endpoint POST /ask y devuelve el JSON de respuesta.
    Lanza RequestException si hay error de red o timeout.
    """
    url     = f"{api_base}/ask"
    payload = {"question": question, "historial": []}
    resp    = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Evaluador principal ────────────────────────────────────────────────────────

def run_evaluation(
    api_base: str,
    delay: float,
    category_filter: str | None,
) -> None:

    # -- Cargar dataset --
    if not DATASET_PATH.exists():
        print(f"[ERROR] No se encontró el dataset en {DATASET_PATH}")
        sys.exit(1)

    dataset: list[dict] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if category_filter:
        dataset = [d for d in dataset if d.get("category") == category_filter]
        if not dataset:
            print(f"[ERROR] No hay entradas con category='{category_filter}'")
            sys.exit(1)

    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"  Evaluación RAG Baseline — ETSI Informática (UMA)")
    print(f"  API: {api_base}")
    print(f"  Dataset: {total} preguntas" + (f" [filtro: {category_filter}]" if category_filter else ""))
    print(f"  Delay entre peticiones: {delay}s")
    print(f"{'='*60}\n")

    results: list[dict] = []

    # Contadores para el resumen final
    counters = {
        "total":          total,
        "success":        0,    # respuesta obtenida (no timeout/error)
        "no_context":     0,    # bot dijo "no encontré información"
        "answered":       0,    # bot dio una respuesta real
        "api_errors":     0,    # error HTTP o de red
    }
    keyword_scores: list[float] = []

    for idx, entry in enumerate(dataset, start=1):
        q_id       = entry.get("id", idx)
        question   = entry["question"]
        ground_truth = entry["ground_truth"]
        category   = entry.get("category", "unknown")

        print(f"[{idx:02d}/{total}] (id={q_id}) {question[:70]}{'...' if len(question)>70 else ''}")

        start_ts = time.time()
        try:
            api_resp = call_ask(api_base, question)
            elapsed  = round(time.time() - start_ts, 2)

            answer   = api_resp.get("answer", "")
            sources  = api_resp.get("sources", [])
            log_id   = api_resp.get("log_id")

            # -- Métricas heurísticas --
            no_ctx   = _no_context_response(answer)
            kw_score = _keyword_overlap(answer, ground_truth)
            ans_len  = _answer_length(answer)

            # Actualizar contadores
            counters["success"] += 1
            if no_ctx:
                counters["no_context"] += 1
                status_label = "SIN CONTEXTO"
            else:
                counters["answered"] += 1
                keyword_scores.append(kw_score)
                status_label = f"OK  kw={kw_score:.2f}  words={ans_len}"

            print(f"         → {status_label}  ({elapsed}s)  log_id={log_id}")

            results.append({
                "id":           q_id,
                "category":     category,
                "question":     question,
                "ground_truth": ground_truth,
                # -- Respuesta del bot --
                "answer":       answer,
                "sources":      sources,
                "log_id":       log_id,
                # -- Métricas heurísticas --
                "metrics": {
                    "no_context_response": no_ctx,
                    "keyword_overlap":     kw_score,
                    "answer_length_words": ans_len,
                    "latency_seconds":     elapsed,
                },
                # -- Placeholder para métricas RAGAS (añadir más adelante) --
                "ragas": {
                    "faithfulness":       None,
                    "answer_relevancy":   None,
                    "context_recall":     None,
                    "context_precision":  None,
                },
            })

        except requests.exceptions.ConnectionError:
            elapsed = round(time.time() - start_ts, 2)
            print(f"         → ERROR  No se pudo conectar con {api_base} — ¿está el servidor arriba?")
            counters["api_errors"] += 1
            results.append({
                "id":           q_id,
                "category":     category,
                "question":     question,
                "ground_truth": ground_truth,
                "answer":       None,
                "sources":      [],
                "log_id":       None,
                "metrics": {
                    "no_context_response": None,
                    "keyword_overlap":     None,
                    "answer_length_words": None,
                    "latency_seconds":     elapsed,
                },
                "ragas":        {},
                "error":        "ConnectionError",
            })

        except requests.exceptions.HTTPError as exc:
            elapsed = round(time.time() - start_ts, 2)
            print(f"         → ERROR HTTP {exc.response.status_code}: {exc.response.text[:120]}")
            counters["api_errors"] += 1
            results.append({
                "id":           q_id,
                "category":     category,
                "question":     question,
                "ground_truth": ground_truth,
                "answer":       None,
                "sources":      [],
                "log_id":       None,
                "metrics": {
                    "no_context_response": None,
                    "keyword_overlap":     None,
                    "answer_length_words": None,
                    "latency_seconds":     elapsed,
                },
                "ragas":        {},
                "error":        f"HTTP {exc.response.status_code}",
            })

        except Exception as exc:
            elapsed = round(time.time() - start_ts, 2)
            print(f"         → ERROR inesperado: {exc}")
            counters["api_errors"] += 1
            results.append({
                "id":           q_id,
                "category":     category,
                "question":     question,
                "ground_truth": ground_truth,
                "answer":       None,
                "sources":      [],
                "log_id":       None,
                "metrics": {},
                "ragas":        {},
                "error":        str(exc),
            })

        # Delay entre peticiones (evita saturar la API de Groq)
        if idx < total:
            time.sleep(delay)

    # ── Guardar resultados ─────────────────────────────────────────────────────

    avg_kw = round(sum(keyword_scores) / len(keyword_scores), 3) if keyword_scores else 0.0

    output = {
        "meta": {
            "timestamp":      datetime.now().isoformat(),
            "api_base":       api_base,
            "dataset_path":   str(DATASET_PATH),
            "category_filter": category_filter,
            "delay_seconds":  delay,
        },
        "summary": {
            **counters,
            "answer_rate":         round(counters["answered"]    / total, 3),
            "no_context_rate":     round(counters["no_context"]  / total, 3),
            "avg_keyword_overlap": avg_kw,
        },
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Resumen en consola ─────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Total preguntas evaluadas : {total}")
    print(f"  Respondidas (con contexto): {counters['answered']}  ({counters['answered']/total*100:.1f}%)")
    print(f"  Sin contexto (RAG fallido): {counters['no_context']}  ({counters['no_context']/total*100:.1f}%)")
    print(f"  Errores de API            : {counters['api_errors']}")
    print(f"  Keyword Overlap medio     : {avg_kw:.3f}  (0=peor, 1=mejor)")
    print(f"\n  Resultados guardados en   : {RESULTS_PATH}")
    print(f"{'='*60}\n")

    # ── Desglose por categoría ─────────────────────────────────────────────────
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "answered": 0, "kw_scores": []}
        categories[cat]["total"] += 1
        m = r.get("metrics", {})
        if m.get("no_context_response") is False:
            categories[cat]["answered"] += 1
        if m.get("keyword_overlap") is not None:
            categories[cat]["kw_scores"].append(m["keyword_overlap"])

    print("  Desglose por categoría:")
    print(f"  {'Categoría':<35} {'Respondidas':>12} {'KW Overlap':>12}")
    print(f"  {'-'*60}")
    for cat, stats in sorted(categories.items()):
        answered_pct = f"{stats['answered']}/{stats['total']}"
        avg_cat_kw   = (
            round(sum(stats["kw_scores"]) / len(stats["kw_scores"]), 3)
            if stats["kw_scores"] else 0.0
        )
        print(f"  {cat:<35} {answered_pct:>12} {avg_cat_kw:>12.3f}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación baseline del sistema RAG de la ETSI Informática."
    )
    parser.add_argument(
        "--api",
        default=DEFAULT_API,
        help=f"URL base de la API FastAPI (default: {DEFAULT_API})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Segundos de espera entre peticiones para no saturar Groq (default: 1.0)",
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=[
            "datos_maestros",
            "normativa_permanencia",
            "tramites_administrativos",
            "preguntas_conflicto",
        ],
        help="Evaluar solo una categoría del dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        api_base=args.api,
        delay=args.delay,
        category_filter=args.category,
    )
