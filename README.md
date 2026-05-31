# Asistente Virtual ETSI Informática (UMA)

Asistente conversacional con RAG híbrido para la Escuela Técnica Superior de Ingeniería Informática de la Universidad de Málaga. Combina búsqueda vectorial (ChromaDB), recuperación avanzada (multi-query, HyDE, re-ranking) y un grafo de conocimiento en Neo4j con titulaciones, planes de estudio y normativa estructurada.

## Requisitos

- Python 3.11+
- Docker Desktop (opcional, recomendado para despliegue con Neo4j)
- Claves de API:
  - **Groq** — obligatoria (LLM de producción)
  - **Anthropic** — solo para scripts de evaluación RAGAS

## Instalación local

```bash
# Clonar o descargar el repositorio y situarse en la raíz del proyecto
cd AsistenteVirtual_ETSII

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Configurar variables de entorno
copy .env.example .env           # Windows
# cp .env.example .env           # Linux / macOS
# Editar .env con las claves reales
```

## Datos e ingesta

El asistente lee la colección ChromaDB `etsi_hibrida`. Si `data/` está vacío, ejecutar el pipeline completo desde la raíz del proyecto:

```bash
# 1. Descargar PDFs y páginas web de la ETSI
python -m scripts.ingestion.crawl_etsi --reset

# 2. Convertir PDFs a Markdown
python -m scripts.ingestion.pdf_to_markdown

# 3. Indexar Markdown (PDF + web) en ChromaDB
python -m src.ingestion.ingest_markdown --reset

# 4. Ingestar hechos curados (datos_maestros.txt, un chunk por línea)
python -m scripts.ingestion.ingest_datos_maestros

# 5. Poblar el grafo Neo4j (requiere Neo4j en marcha)
python -m src.ingestion.build_graph --reset
```

Los pasos 1–4 pueden ejecutarse sin Neo4j. El paso 5 es independiente del vector store.

**Estructura de `data/`:**

| Ruta | Contenido |
|---|---|
| `data/raw/` | PDFs descargados, `datos_maestros.txt`, mapas de fuentes JSON |
| `data/processed/` | Markdown generado (PDFs y webs) |
| `data/chroma_db/` | Base vectorial persistente |
| `data/chatlogs.db` | Logs de interacciones y feedback |
| `data/eval/` | Dataset de evaluación y resultados RAGAS |

## Ejecución con Docker

```bash
docker compose up --build
```

Servicios levantados:

| Servicio | Puerto | Función |
|---|---|---|
| `api` | 8000 | FastAPI (chat, admin, estáticos) |
| `neo4j` | 7474 / 7687 | Grafo de conocimiento |

El contenedor `api` monta `./data` como volumen. Los scripts de ingesta se ejecutan en el host, no dentro del contenedor.

Tras el primer arranque con Docker, poblar el grafo desde el host (Neo4j accesible en `localhost:7687`):

```bash
python -m src.ingestion.build_graph --reset
```

## Ejecución local del servidor

Con Neo4j opcional (si no hay conexión, el sistema opera solo con búsqueda vectorial):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Interfaces

| URL | Descripción |
|---|---|
| `http://localhost:8000/docs` | Documentación interactiva de la API |
| `http://localhost:8000/static/demo/demo.html` | Página demo con instrucciones de embed |
| `http://localhost:8000/static/admin/login.html` | Panel de administración |
| `http://localhost:7474` | Neo4j Browser (usuario `neo4j`) |

### Widget embebible

Incluir en cualquier página:

```html
<script src="http://localhost:8000/static/widget/embed.js"></script>
```

Opcional: definir `window.ETSI_WIDGET_API` antes del script para apuntar a otra URL base.

## API principal

### `POST /ask`

```json
{
  "question": "¿Cuántos créditos ECTS tiene el Grado en Ingeniería del Software?",
  "historial": []
}
```

Respuesta:

```json
{
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "log_id": 42
}
```

### Otros endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/feedback` | Enviar valoración (0/1) de una respuesta |
| `POST` | `/login` | Autenticación admin (JWT) |
| `GET` | `/admin/stats` | Estadísticas agregadas |
| `GET` | `/admin/logs` | Logs paginados |
| `GET/PUT` | `/admin/config` | Prompt y modelo del LLM |

## Scripts de evaluación

Requieren `ANTHROPIC_API_KEY` y la API en marcha.

```bash
# Generar respuestas contra /ask (modo híbrido completo)
python -m scripts.evaluation.run_baseline_eval

# Evaluar un fichero de resultados con RAGAS
python -m scripts.evaluation.evaluate_ragas --input data/eval/results_hybrid_v2.json

# Comparativa RAG plano vs GraphRAG
python -m scripts.evaluation.comparative_evaluation \
  --plain data/eval/results_plain_rag.json \
  --graph data/eval/results_graph_rag.json
```

Para la comparativa plano vs grafo: detener Neo4j y reiniciar la API para generar respuestas sin grafo; volver a levantar Neo4j para el modo GraphRAG.

## Variables de entorno

Ver `.env.example`. Resumen:

| Variable | Uso |
|---|---|
| `GROQ_API_KEY` | LLM principal |
| `ANTHROPIC_API_KEY` | Juez RAGAS (evaluación) |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Grafo Neo4j |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | Panel admin |
| `JWT_SECRET_KEY` | Firma de tokens |
| `BASE_URL` | URL base para enlaces a fuentes |
| `ALLOWED_ORIGINS` | Orígenes CORS permitidos |

## Estructura del proyecto

```
src/
  api/           FastAPI y endpoints
  core/          Router, retriever, LLM, grafo
  database/      ChromaDB y SQLite
  ingestion/     ingest_markdown, build_graph
  frontend/      Widget, admin, demo
scripts/
  ingestion/     crawl, pdf_to_markdown, datos_maestros
  evaluation/    baseline, RAGAS, comparativa
tests/
  debug_search.py   Utilidad de depuración de ChromaDB
```

## Arquitectura de recuperación

```
Pregunta → Router (intención + seguridad)
         → AdvancedRetriever (ChromaDB: multi-query + HyDE + rerank)
         → GraphRetriever (Neo4j: titulaciones, planes, normativa)
         → ChatEngine (Groq LLM)
```

Si Neo4j no está disponible, `GraphRetriever` se desactiva sin interrumpir el flujo.
