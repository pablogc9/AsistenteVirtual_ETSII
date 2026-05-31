#!/usr/bin/env python3
"""
Evaluación comparativa científica entre RAG Plano y GraphRAG.

Ejecuta la librería RAGAS (faithfulness + answer_correctness) sobre ambos
conjuntos de respuestas y genera:

  1. Tabla comparativa por categoría impresa en consola.
  2. Análisis de 'Éxitos Críticos del Grafo': ítems donde RAG Plano falló
     (correctness < 0.3) y GraphRAG acertó (correctness > 0.8).
  3. Informe completo guardado en data/eval/final_comparison_report.json.

Nota metodológica (faithfulness):
  Al no disponer de los fragmentos reales recuperados, se usa ground_truth
  como contexto proxy. La comparación sigue siendo válida porque el sesgo
  afecta a ambos sistemas por igual.

Uso (ejecutar desde la raíz del proyecto):
    python -m scripts.evaluation.comparative_evaluation
    python -m scripts.evaluation.comparative_evaluation --limit 10     # prueba rápida
    python -m scripts.evaluation.comparative_evaluation --plain data/eval/results_plain_rag.json
                                                        --graph data/eval/results_graph_rag.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Forzar UTF-8 en consola Windows (evita UnicodeEncodeError con tildes/símbolos)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── RAGAS imports ──────────────────────────────────────────────────────────────
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import faithfulness, answer_correctness
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    from ragas.run_config import RunConfig
except ImportError as e:
    print(f"[ERROR] No se pudo importar RAGAS: {e}")
    print("        Instálalo con: pip install 'ragas>=0.4.0'")
    sys.exit(1)

from anthropic import Anthropic

# ── Rutas por defecto ──────────────────────────────────────────────────────────
DEFAULT_PLAIN = Path("data/eval/results_plain_rag.json")
DEFAULT_GRAPH = Path("data/eval/results_graph_rag.json")
DEFAULT_OUTPUT = Path("data/eval/final_comparison_report.json")

EVAL_MODEL = "claude-sonnet-4-20250514"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Umbrales para análisis de Éxitos Críticos
CRITICAL_PLAIN_MAX = 0.3   # RAG Plano tenía que fallar aquí o por debajo
CRITICAL_GRAPH_MIN = 0.8   # GraphRAG tenía que acertar aquí o por encima

# Nombres de categoría para mostrar en la tabla
CATEGORY_LABELS = {
    "datos_maestros":            "Datos Maestros",
    "normativa_permanencia":     "Normativa y Permanencia",
    "tramites_administrativos":  "Trámites Administrativos",
    "preguntas_conflicto":       "Preguntas de Conflicto",
}

# ── Configuración del evaluador ────────────────────────────────────────────────

def _build_evaluator():
    """
    Inicializa el LLM juez (Claude Sonnet vía Anthropic) y los embeddings multilingües.

    RAGAS 0.4.x requiere InstructorLLM. El cliente nativo de Anthropic tiene
    .messages y funciona directamente con llm_factory(provider='anthropic').
    """
    print(f"  Cargando modelo juez: {EVAL_MODEL}...")
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    llm = llm_factory(EVAL_MODEL, provider="anthropic", client=anthropic_client, max_tokens=8096)
    emb = RagasHFEmbeddings(model=EMBED_MODEL)

    faithfulness.llm         = llm
    answer_correctness.llm   = llm
    answer_correctness.embeddings = emb

    run_cfg = RunConfig(max_retries=3, max_wait=300, timeout=300, seed=42)
    return faithfulness, answer_correctness, run_cfg


# ── Carga y validación de datos ────────────────────────────────────────────────

def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Archivo no encontrado: {path}")
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data) if isinstance(data, dict) else data
    if not results:
        print(f"[ERROR] El archivo {path} no contiene resultados.")
        sys.exit(1)
    return results


def _build_dataset(results: list[dict], limit: int | None) -> EvaluationDataset:
    """
    Construye un EvaluationDataset de RAGAS.
    Usa ground_truth como retrieved_contexts (proxy para faithfulness).
    Omite ítems con answer=None.
    """
    items = results[:limit] if limit else results
    samples = []
    for item in items:
        if item.get("answer") is None:
            continue
        samples.append(
            SingleTurnSample(
                user_input         = item["question"],
                response           = item["answer"],
                retrieved_contexts = [item["ground_truth"]],
                reference          = item["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


# ── Evaluación de un dataset ───────────────────────────────────────────────────

def _evaluate_dataset(
    label:        str,
    dataset:      EvaluationDataset,
    faith_metric,
    corr_metric,
    run_cfg,
) -> list[dict]:
    """
    Ejecuta RAGAS sobre un dataset y devuelve una lista de dicts con
    {faithfulness, answer_correctness} por ítem.
    """
    n = len(dataset.samples)
    est_min = n * 4 // 60
    est_max = n * 8 // 60 + 1
    print(f"\n  [{label}] Evaluando {n} muestras (~{est_min}-{est_max} min)...")

    t0 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=[faith_metric, corr_metric],
        run_config=run_cfg,
        raise_exceptions=False,
        batch_size=1,
    )
    elapsed = round(time.time() - t0, 1)
    print(f"  [{label}] Completado en {elapsed}s")

    df = result.to_pandas()
    scores = []
    for _, row in df.iterrows():
        f = float(row.get("faithfulness",      float("nan")))
        c = float(row.get("answer_correctness", float("nan")))
        scores.append({
            "faithfulness":       None if f != f else round(f, 4),
            "answer_correctness": None if c != c else round(c, 4),
        })
    return scores


# ── Fusión de resultados ───────────────────────────────────────────────────────

def _merge_scores(
    results: list[dict],
    scores:  list[dict],
    limit:   int | None,
) -> list[dict]:
    """
    Combina la lista de resultados del JSON con los scores de RAGAS,
    omitiendo los ítems que fueron saltados (answer=None).
    """
    items    = results[:limit] if limit else results
    valid    = [r for r in items if r.get("answer") is not None]
    merged   = []
    score_idx = 0
    for item in valid:
        s = scores[score_idx] if score_idx < len(scores) else {}
        merged.append({
            "id":                  item["id"],
            "category":            item["category"],
            "question":            item["question"],
            "ground_truth":        item["ground_truth"],
            "answer":              item["answer"],
            "faithfulness":        s.get("faithfulness"),
            "answer_correctness":  s.get("answer_correctness"),
        })
        score_idx += 1
    return merged


# ── Estadísticas ───────────────────────────────────────────────────────────────

def _avg(values: list[float]) -> float | None:
    v = [x for x in values if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def _compute_stats(merged: list[dict]) -> dict[str, dict]:
    """Calcula media de faithfulness y answer_correctness por categoría + global."""
    cats: dict[str, dict] = {}
    for item in merged:
        cat = item["category"]
        if cat not in cats:
            cats[cat] = {"faithfulness": [], "answer_correctness": []}
        cats[cat]["faithfulness"].append(item["faithfulness"])
        cats[cat]["answer_correctness"].append(item["answer_correctness"])

    stats: dict[str, dict] = {}
    all_f, all_c = [], []
    for cat, lists in cats.items():
        stats[cat] = {
            "n":                       len(lists["faithfulness"]),
            "faithfulness_mean":       _avg(lists["faithfulness"]),
            "answer_correctness_mean": _avg(lists["answer_correctness"]),
        }
        all_f += lists["faithfulness"]
        all_c += lists["answer_correctness"]

    stats["__global__"] = {
        "n":                       len(merged),
        "faithfulness_mean":       _avg(all_f),
        "answer_correctness_mean": _avg(all_c),
    }
    return stats


# ── Análisis de Éxitos Críticos ────────────────────────────────────────────────

def _critical_successes(
    plain_merged: list[dict],
    graph_merged: list[dict],
) -> list[dict]:
    """
    Identifica ítems donde:
      - RAG Plano: answer_correctness < CRITICAL_PLAIN_MAX (falló)
      - GraphRAG : answer_correctness > CRITICAL_GRAPH_MIN (acertó)
    """
    plain_map = {r["id"]: r for r in plain_merged}
    graph_map = {r["id"]: r for r in graph_merged}

    successes = []
    for item_id, graph_item in graph_map.items():
        plain_item = plain_map.get(item_id)
        if plain_item is None:
            continue
        pc = plain_item.get("answer_correctness")
        gc = graph_item.get("answer_correctness")
        if pc is None or gc is None:
            continue
        if pc < CRITICAL_PLAIN_MAX and gc > CRITICAL_GRAPH_MIN:
            successes.append({
                "id":                     item_id,
                "category":               graph_item["category"],
                "question":               graph_item["question"],
                "ground_truth":           graph_item["ground_truth"],
                "plain_answer":           plain_item["answer"],
                "graph_answer":           graph_item["answer"],
                "plain_correctness":      round(pc, 4),
                "graph_correctness":      round(gc, 4),
                "improvement":            round(gc - pc, 4),
            })

    return sorted(successes, key=lambda x: x["improvement"], reverse=True)


# ── Impresión del informe ──────────────────────────────────────────────────────

def _print_report(
    plain_stats:      dict,
    graph_stats:      dict,
    critical:         list[dict],
) -> None:

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    def pct(plain_v, graph_v):
        if plain_v is None or graph_v is None or plain_v == 0:
            return "   N/A"
        delta = (graph_v - plain_v) / plain_v * 100
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    W = 72
    print(f"\n{'═'*W}")
    print(f"  INFORME COMPARATIVO: RAG Plano  vs.  GraphRAG (ETSI Informática)")
    print(f"  Modelo juez: {EVAL_MODEL}")
    print(f"{'═'*W}")

    # ── Tabla de faithfulness ──────────────────────────────────────────────
    print(f"\n  {'FAITHFULNESS':─<68}")
    print(f"  {'Categoría':<32} {'RAG Plano':>10} {'GraphRAG':>10} {'Mejora':>8}")
    print(f"  {'─'*62}")

    cats_ordered = [
        "datos_maestros",
        "normativa_permanencia",
        "tramites_administrativos",
        "preguntas_conflicto",
        "__global__",
    ]
    for cat in cats_ordered:
        label = "  ── GLOBAL ──" if cat == "__global__" else CATEGORY_LABELS.get(cat, cat)
        ps    = plain_stats.get(cat, {})
        gs    = graph_stats.get(cat, {})
        pv    = ps.get("faithfulness_mean")
        gv    = gs.get("faithfulness_mean")
        if cat == "__global__":
            print(f"  {'─'*62}")
        print(f"  {label:<32} {fmt(pv):>10} {fmt(gv):>10} {pct(pv,gv):>8}")

    # ── Tabla de answer_correctness ────────────────────────────────────────
    print(f"\n  {'ANSWER CORRECTNESS':─<68}")
    print(f"  {'Categoría':<32} {'RAG Plano':>10} {'GraphRAG':>10} {'Mejora':>8}")
    print(f"  {'─'*62}")

    for cat in cats_ordered:
        label = "  ── GLOBAL ──" if cat == "__global__" else CATEGORY_LABELS.get(cat, cat)
        ps    = plain_stats.get(cat, {})
        gs    = graph_stats.get(cat, {})
        pv    = ps.get("answer_correctness_mean")
        gv    = gs.get("answer_correctness_mean")
        if cat == "__global__":
            print(f"  {'─'*62}")
        print(f"  {label:<32} {fmt(pv):>10} {fmt(gv):>10} {pct(pv,gv):>8}")

    # ── Éxitos Críticos ────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  ÉXITOS CRÍTICOS DEL GRAFO  "
          f"(plain < {CRITICAL_PLAIN_MAX} → graph > {CRITICAL_GRAPH_MIN})")
    print(f"{'═'*W}")

    if not critical:
        print(f"  No se encontraron éxitos críticos con los umbrales actuales.")
    else:
        print(f"  {len(critical)} caso(s) encontrado(s):\n")
        for i, item in enumerate(critical, 1):
            label = CATEGORY_LABELS.get(item["category"], item["category"])
            print(f"  [{i}] ID={item['id']}  [{label}]")
            print(f"      Pregunta  : {item['question'][:70]}")
            print(f"      Plano     : {item['plain_correctness']:.3f}  → "
                  f"'{item['plain_answer'][:60]}...'")
            print(f"      GraphRAG  : {item['graph_correctness']:.3f}  → "
                  f"'{item['graph_answer'][:60]}...'")
            print(f"      Mejora    : +{item['improvement']:.3f}")
            print()

    print(f"{'═'*W}\n")


# ── Punto de entrada ───────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    print(f"\n{'='*72}")
    print(f"  EVALUACIÓN COMPARATIVA RAGAS — ETSI Informática (UMA)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}")

    # ── 1. Cargar datos ────────────────────────────────────────────────────
    print(f"\n  Cargando archivos...")
    plain_results = _load_results(args.plain)
    graph_results = _load_results(args.graph)
    print(f"  RAG Plano : {len(plain_results)} ítems  ({args.plain.name})")
    print(f"  GraphRAG  : {len(graph_results)} ítems  ({args.graph.name})")

    # ── 2. Construir datasets RAGAS ────────────────────────────────────────
    plain_dataset = _build_dataset(plain_results, args.limit)
    graph_dataset = _build_dataset(graph_results, args.limit)
    print(f"\n  Muestras válidas — RAG Plano: {len(plain_dataset.samples)} | "
          f"GraphRAG: {len(graph_dataset.samples)}")

    # ── 3. Inicializar evaluador (una sola vez para ambos) ─────────────────
    faith_metric, corr_metric, run_cfg = _build_evaluator()

    # ── 4. Evaluar RAG Plano ───────────────────────────────────────────────
    plain_scores = _evaluate_dataset(
        "RAG Plano", plain_dataset, faith_metric, corr_metric, run_cfg
    )

    # Pequeña pausa para no acumular contexto de rate-limit entre bloques
    print("  Pausa de 10s entre evaluaciones...")
    time.sleep(10)

    # ── 5. Evaluar GraphRAG ────────────────────────────────────────────────
    graph_scores = _evaluate_dataset(
        "GraphRAG", graph_dataset, faith_metric, corr_metric, run_cfg
    )

    # ── 6. Fusionar scores con metadatos originales ────────────────────────
    plain_merged = _merge_scores(plain_results, plain_scores, args.limit)
    graph_merged = _merge_scores(graph_results, graph_scores, args.limit)

    # ── 7. Estadísticas por categoría ─────────────────────────────────────
    plain_stats = _compute_stats(plain_merged)
    graph_stats = _compute_stats(graph_merged)

    # ── 8. Éxitos Críticos del Grafo ───────────────────────────────────────
    critical = _critical_successes(plain_merged, graph_merged)

    # ── 9. Imprimir informe en consola ─────────────────────────────────────
    _print_report(plain_stats, graph_stats, critical)

    # ── 10. Guardar informe JSON ────────────────────────────────────────────
    if not args.no_save:
        _save_report(
            args,
            plain_merged, graph_merged,
            plain_stats, graph_stats,
            critical,
        )


def _save_report(
    args,
    plain_merged: list[dict],
    graph_merged: list[dict],
    plain_stats:  dict,
    graph_stats:  dict,
    critical:     list[dict],
) -> None:

    # Construir tabla comparativa plana para el JSON
    comparison_table: list[dict] = []
    cats_ordered = [
        "datos_maestros",
        "normativa_permanencia",
        "tramites_administrativos",
        "preguntas_conflicto",
        "__global__",
    ]
    for cat in cats_ordered:
        ps = plain_stats.get(cat, {})
        gs = graph_stats.get(cat, {})
        for metric in ("faithfulness_mean", "answer_correctness_mean"):
            pv = ps.get(metric)
            gv = gs.get(metric)
            delta_pct = None
            if pv is not None and gv is not None and pv != 0:
                delta_pct = round((gv - pv) / pv * 100, 2)
            comparison_table.append({
                "category": cat,
                "metric":   metric.replace("_mean", ""),
                "plain_rag": pv,
                "graph_rag": gv,
                "improvement_pct": delta_pct,
            })

    report = {
        "meta": {
            "timestamp":             datetime.now().isoformat(),
            "llm_evaluator":         EVAL_MODEL,
            "embedding_model":       EMBED_MODEL,
            "plain_rag_file":        str(args.plain),
            "graph_rag_file":        str(args.graph),
            "n_plain_evaluated":     len(plain_merged),
            "n_graph_evaluated":     len(graph_merged),
            "faithfulness_proxy":    "ground_truth (retrieved_contexts no disponibles)",
            "critical_plain_max":    CRITICAL_PLAIN_MAX,
            "critical_graph_min":    CRITICAL_GRAPH_MIN,
        },
        "comparison_table": comparison_table,
        "critical_successes_of_graph": critical,
        "per_item": {
            "plain_rag": plain_merged,
            "graph_rag": graph_merged,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Informe guardado en: {args.output}")

    # También actualiza los campos ragas.* en los archivos fuente
    _patch_source(args.plain, plain_merged)
    _patch_source(args.graph, graph_merged)


def _patch_source(path: Path, merged: list[dict]) -> None:
    """Escribe los scores RAGAS calculados en los campos ragas.* del JSON original."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        score_map = {r["id"]: r for r in merged}
        for item in data.get("results", []):
            if item.get("id") in score_map:
                s = score_map[item["id"]]
                item["ragas"]["faithfulness"]       = s.get("faithfulness")
                item["ragas"]["answer_correctness"] = s.get("answer_correctness")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Scores RAGAS escritos en {path.name}")
    except Exception as exc:
        print(f"  [WARN] No se pudo actualizar {path.name}: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación comparativa RAGAS: RAG Plano vs. GraphRAG."
    )
    p.add_argument(
        "--plain", type=Path, default=DEFAULT_PLAIN,
        help=f"Resultados del RAG Plano (default: {DEFAULT_PLAIN})",
    )
    p.add_argument(
        "--graph", type=Path, default=DEFAULT_GRAPH,
        help=f"Resultados del GraphRAG (default: {DEFAULT_GRAPH})",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Archivo de salida (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limitar a N preguntas por sistema (útil para prueba rápida)",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Solo imprimir en consola, sin guardar archivos",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
