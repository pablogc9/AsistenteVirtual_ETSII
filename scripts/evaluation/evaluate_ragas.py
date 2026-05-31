#!/usr/bin/env python3
"""
Evalúa el sistema RAG con métricas científicas usando la librería RAGAS.

Métricas calculadas:
  · Faithfulness       – ¿Cada afirmación de la respuesta está respaldada por el contexto?
                         Rango: 0.0 (ninguna) → 1.0 (todas). Ideal: > 0.75
                         ⚠ Nota metodológica: al no disponer de los fragmentos reales
                           recuperados por ChromaDB en el JSON, se usa el `ground_truth`
                           como contexto proxy. Esto mide "¿se basa la respuesta en los
                           hechos conocidos?" — una aproximación válida para baseline.

  · Answer Correctness – ¿Es la respuesta factualmente correcta y completa respecto al GT?
                         Combina similitud semántica (embeddings) + precisión factual (LLM).
                         Rango: 0.0 → 1.0. Ideal: > 0.70

Uso (ejecutar desde la raíz del proyecto):
    python -m scripts.evaluation.evaluate_ragas                        # baseline completo
    python -m scripts.evaluation.evaluate_ragas --input data/eval/graphrag_results.json
    python -m scripts.evaluation.evaluate_ragas --category preguntas_conflicto
    python -m scripts.evaluation.evaluate_ragas --limit 10             # primeras 10
    python -m scripts.evaluation.evaluate_ragas --no-save             # no guarda

Requisitos:
    pip install "ragas>=0.4.0"
    ANTHROPIC_API_KEY en el archivo .env
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Imports RAGAS 0.4.x ───────────────────────────────────────────────────────
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

# ── Rutas por defecto ─────────────────────────────────────────────────────────

DEFAULT_INPUT  = Path("data/eval/results_hybrid_v2.json")
DEFAULT_OUTPUT = Path("data/eval/ragas_report.json")

# Señal de que el bot no encontró contexto
NO_CONTEXT_SIGNAL = "no he encontrado información"

# ── Configuración ──────────────────────────────────────────────────────────────

def _build_llm_and_embeddings(llm_model: str):
    """
    Configura el LLM evaluador (Claude Sonnet vía Anthropic) y los embeddings.

    El cliente nativo de Anthropic tiene .messages y funciona directamente
    con llm_factory(provider='anthropic') en RAGAS 0.4.x.
    """
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    evaluator_llm        = llm_factory(llm_model, provider="anthropic", client=anthropic_client, max_tokens=8096)
    evaluator_embeddings = RagasHFEmbeddings(model="paraphrase-multilingual-MiniLM-L12-v2")
    return evaluator_llm, evaluator_embeddings


# ── Construcción del dataset ──────────────────────────────────────────────────

def _build_samples(
    results:         list[dict],
    category_filter: str | None,
    limit:           int | None,
) -> tuple[list[SingleTurnSample], list[dict]]:
    """
    Convierte los resultados del JSON en objetos SingleTurnSample de RAGAS.

    Devuelve:
        samples   – lista de muestras para RAGAS (solo ítems con respuesta válida)
        skipped   – ítems descartados (sin respuesta o error de API)
    """
    samples: list[SingleTurnSample] = []
    skipped: list[dict]             = []

    filtered = results
    if category_filter:
        filtered = [r for r in filtered if r.get("category") == category_filter]

    if limit:
        filtered = filtered[:limit]

    for item in filtered:
        answer = item.get("answer")

        # Descartar ítems sin respuesta (errores de API durante la evaluación)
        if answer is None:
            skipped.append({**item, "_skip_reason": "answer_is_null"})
            continue

        # Los ítems "sin contexto" se evalúan también — su score bajo es información útil
        question     = item["question"]
        ground_truth = item["ground_truth"]

        # Contexto proxy para faithfulness:
        # Usamos ground_truth porque el JSON no almacena los fragmentos recuperados.
        # Esto mide "¿la respuesta contiene solo afirmaciones que están en los hechos
        # conocidos?" — válido para comparar sistemas entre sí.
        proxy_context = [ground_truth]

        samples.append(
            SingleTurnSample(
                user_input         = question,
                response           = answer,
                retrieved_contexts = proxy_context,
                reference          = ground_truth,
            )
        )

    return samples, skipped


# ── Evaluación principal ──────────────────────────────────────────────────────

def run_evaluation(
    input_path:      Path,
    output_path:     Path,
    llm_model:       str,
    category_filter: str | None,
    limit:           int | None,
    save_results:    bool,
    update_source:   bool,
) -> None:

    # ── 1. Cargar resultados ────────────────────────────────────────────────

    if not input_path.exists():
        print(f"[ERROR] No se encontró {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    results: list[dict] = data.get("results", data) if isinstance(data, dict) else data

    print(f"\n{'='*65}")
    print(f"  Evaluación RAGAS — ETSI Informática (UMA)")
    print(f"  Modelo juez : {llm_model}")
    print(f"  Archivo     : {input_path}")
    print(f"  Filtro      : {category_filter or 'todas las categorías'}")
    print(f"  Límite      : {limit or 'sin límite'}")
    print(f"{'='*65}\n")

    # ── 2. Construir samples ────────────────────────────────────────────────

    samples, skipped = _build_samples(results, category_filter, limit)

    if not samples:
        print("[ERROR] No hay muestras válidas para evaluar.")
        sys.exit(1)

    print(f"  Muestras válidas : {len(samples)}")
    print(f"  Descartadas      : {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    - id={s.get('id')} ({s.get('_skip_reason', 'unknown')})")
    print()

    dataset = EvaluationDataset(samples=samples)

    # ── 3. Configurar LLM y embeddings ─────────────────────────────────────

    print(f"  Cargando modelo evaluador ({llm_model})...")
    evaluator_llm, evaluator_embeddings = _build_llm_and_embeddings(llm_model)

    # Asignar LLM/embeddings a los singletons de ragas.metrics
    faithfulness.llm              = evaluator_llm
    answer_correctness.llm        = evaluator_llm
    answer_correctness.embeddings = evaluator_embeddings

    faithfulness_metric       = faithfulness
    answer_correctness_metric = answer_correctness

    run_cfg = RunConfig(
        max_retries=3,
        max_wait=300,
        timeout=300,
        seed=42,
    )

    # ── 4. Ejecutar evaluación ──────────────────────────────────────────────

    print(f"  Evaluando {len(samples)} muestras...")
    print(f"  (Esto puede tardar {len(samples) * 4 // 60 + 1}–{len(samples) * 8 // 60 + 2} min "
          f"con {llm_model})\n")

    t0 = time.time()
    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_correctness_metric],
            run_config=run_cfg,
            raise_exceptions=False,
            batch_size=1,
        )
    except Exception as exc:
        print(f"\n[ERROR] La evaluación RAGAS falló: {exc}")
        sys.exit(1)

    elapsed = round(time.time() - t0, 1)
    print(f"\n  Evaluación completada en {elapsed}s")

    # ── 5. Extraer resultados por ítem ──────────────────────────────────────

    df = eval_result.to_pandas()

    # Mapear scores de vuelta a los ítems originales por posición
    filtered_items: list[dict] = results
    if category_filter:
        filtered_items = [r for r in results if r.get("category") == category_filter]
    if limit:
        filtered_items = filtered_items[:limit]

    valid_items = [r for r in filtered_items if r.get("answer") is not None]

    per_item_results: list[dict] = []
    for idx, item in enumerate(valid_items):
        if idx >= len(df):
            break
        row = df.iloc[idx]
        faith = float(row.get("faithfulness", float("nan")))
        corr  = float(row.get("answer_correctness", float("nan")))

        per_item_results.append({
            "id":                  item.get("id"),
            "category":            item.get("category"),
            "question":            item["question"],
            "ground_truth":        item["ground_truth"],
            "answer":              item["answer"],
            "faithfulness":        None if faith != faith else round(faith, 4),  # NaN → None
            "answer_correctness":  None if corr  != corr  else round(corr,  4),
            "metrics_heuristic":   item.get("metrics", {}),
        })

    # ── 6. Calcular resumen global y por categoría ──────────────────────────

    mean_faith = eval_result.get("faithfulness",      None)
    mean_corr  = eval_result.get("answer_correctness", None)

    # Resumen por categoría
    cat_stats: dict[str, dict] = {}
    for r in per_item_results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"faithfulness": [], "answer_correctness": []}
        if r["faithfulness"] is not None:
            cat_stats[cat]["faithfulness"].append(r["faithfulness"])
        if r["answer_correctness"] is not None:
            cat_stats[cat]["answer_correctness"].append(r["answer_correctness"])

    cat_summary: dict[str, dict] = {}
    for cat, scores in cat_stats.items():
        f_list = scores["faithfulness"]
        c_list = scores["answer_correctness"]
        cat_summary[cat] = {
            "n":                   len(f_list),
            "faithfulness_mean":   round(sum(f_list) / len(f_list), 4) if f_list else None,
            "answer_correctness_mean": round(sum(c_list) / len(c_list), 4) if c_list else None,
        }

    # ── 7. Imprimir informe ─────────────────────────────────────────────────

    _print_report(mean_faith, mean_corr, cat_summary, per_item_results, elapsed)

    # ── 8. Guardar resultados ───────────────────────────────────────────────

    if save_results:
        report = {
            "meta": {
                "timestamp":      datetime.now().isoformat(),
                "input_file":     str(input_path),
                "llm_evaluator":  llm_model,
                "ragas_version":  _get_ragas_version(),
                "n_evaluated":    len(per_item_results),
                "n_skipped":      len(skipped),
                "elapsed_seconds": elapsed,
                "nota_metodologica": (
                    "faithfulness usa ground_truth como contexto proxy ya que el archivo "
                    "de resultados no almacena los fragmentos recuperados. Mide si las "
                    "afirmaciones de la respuesta están respaldadas por los hechos conocidos."
                ),
            },
            "summary": {
                "faithfulness_mean":         round(mean_faith, 4) if mean_faith is not None else None,
                "answer_correctness_mean":   round(mean_corr,  4) if mean_corr  is not None else None,
                "by_category":               cat_summary,
            },
            "results": per_item_results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Informe RAGAS guardado en: {output_path}")

    # ── 9. (Opcional) Actualizar el archivo fuente con scores RAGAS ─────────
    # Añade los scores directamente al baseline_results.json original para
    # que el Dashboard pueda visualizarlos.

    if update_source and save_results:
        _patch_source_json(input_path, per_item_results)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_ragas_version() -> str:
    try:
        import ragas
        return ragas.__version__
    except Exception:
        return "unknown"


def _print_report(
    mean_faith:       float | None,
    mean_corr:        float | None,
    cat_summary:      dict,
    per_item_results: list[dict],
    elapsed:          float,
) -> None:

    def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
    def bar(v, width=20):
        if v is None: return " " * width
        filled = round(v * width)
        return "█" * filled + "░" * (width - filled)

    print(f"\n{'='*65}")
    print(f"  INFORME RAGAS — RESULTADOS GLOBALES")
    print(f"{'='*65}")
    print(f"  Faithfulness       : {fmt(mean_faith)}  {bar(mean_faith)}")
    print(f"  Answer Correctness : {fmt(mean_corr)}   {bar(mean_corr)}")
    print(f"  Tiempo total       : {elapsed}s")

    print(f"\n  Desglose por categoría:")
    print(f"  {'Categoría':<35} {'N':>4}  {'Faith.':>8}  {'Correct.':>10}")
    print(f"  {'-'*60}")
    for cat, stats in sorted(cat_summary.items()):
        print(
            f"  {cat:<35} {stats['n']:>4}  "
            f"{fmt(stats['faithfulness_mean']):>8}  "
            f"{fmt(stats['answer_correctness_mean']):>10}"
        )

    # Top 5 mejores y peores por answer_correctness
    scored = [r for r in per_item_results if r.get("answer_correctness") is not None]
    if scored:
        by_corr = sorted(scored, key=lambda x: x["answer_correctness"], reverse=True)

        print(f"\n  Top 5 respuestas CORRECTAS (answer_correctness más alto):")
        for r in by_corr[:5]:
            print(f"    [{r['answer_correctness']:.3f}] (id={r['id']}) {r['question'][:60]}")

        print(f"\n  Top 5 respuestas FALLIDAS (answer_correctness más bajo):")
        for r in by_corr[-5:]:
            print(f"    [{r['answer_correctness']:.3f}] (id={r['id']}) {r['question'][:60]}")

    print(f"\n{'='*65}\n")


def _patch_source_json(source_path: Path, per_item_results: list[dict]) -> None:
    """
    Escribe los scores RAGAS en los campos 'ragas.*' del JSON de resultados original,
    para que el Dashboard de administración pueda leerlos.
    """
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
        result_list = data.get("results", [])

        score_map = {r["id"]: r for r in per_item_results}

        for item in result_list:
            item_id = item.get("id")
            if item_id in score_map:
                scored = score_map[item_id]
                item["ragas"] = {
                    "faithfulness":       scored.get("faithfulness"),
                    "answer_correctness": scored.get("answer_correctness"),
                    "answer_relevancy":   None,   # no calculado en este run
                    "context_recall":     None,
                }

        source_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Scores RAGAS escritos en {source_path.name} (campos 'ragas.*')")
    except Exception as exc:
        print(f"  [WARN] No se pudo actualizar {source_path}: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS del sistema RAG de la ETSI Informática."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Archivo de resultados a evaluar (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Archivo donde guardar el informe RAGAS (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Modelo Anthropic a usar como juez evaluador (default: claude-sonnet-4-20250514)",
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
        help="Evaluar solo una categoría (útil para test rápido)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a N preguntas (ej. --limit 5 para prueba rápida)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Solo mostrar resultados en consola, sin guardar archivos",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="No modificar el archivo fuente con los scores RAGAS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        llm_model=args.model,
        category_filter=args.category,
        limit=args.limit,
        save_results=not args.no_save,
        update_source=not args.no_patch,
    )
