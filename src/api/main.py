from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.database.vector_store import VectorStoreManager
from src.core.llm_engine import ChatEngine
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Middleware CORS: debe registrarse antes de definir cualquier ruta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Acepta peticiones desde cualquier dominio
    allow_methods=["*"],      # Acepta GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],      # Acepta cualquier cabecera HTTP
)

app.mount("/pdfs", StaticFiles(directory="data/raw"), name="pdfs")

vector_store = VectorStoreManager()
chat_engine = ChatEngine()

DISTANCE_THRESHOLD = 0.6
NO_CONTEXT_MSG = "Lo siento, no he encontrado información oficial sobre esa pregunta en los documentos de la ETSI Informática."

class HistorialMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    historial: list[HistorialMessage] = []

@app.post("/ask")
def ask(request: AskRequest):
    fragments = vector_store.search(request.question, k=5)

    # Si todos los fragmentos superan el umbral de distancia, la pregunta
    # está demasiado lejos del contenido indexado y no respondemos
    if not fragments or all(frag["distance"] > DISTANCE_THRESHOLD for frag in fragments):
        return {"answer": NO_CONTEXT_MSG, "sources": []}

    print("--- CONTEXTO ENVIADO AL LLM ---")
    for f in fragments:
        print(f["text"][:200]) # Imprime los primeros 200 caracteres de cada chunk
    print("-------------------------------")

    # Construimos fuentes únicas con título y URL/ruta
    seen = set()
    sources = []
    for frag in fragments:
        meta = frag.get("metadata", {})
        raw_source = meta.get("source") or meta.get("source_path", "")
        if not raw_source or raw_source in seen:
            continue
        seen.add(raw_source)

        title = meta.get("title", "")

        # Si es un PDF local, construimos la URL pública
        if raw_source.endswith(".pdf"):
            filename = Path(raw_source).name
            url = f"http://127.0.0.1:8000/pdfs/{filename}"
        else:
            url = raw_source

        sources.append({"title": title, "url": url})

    answer = chat_engine.generate_answer(request.question, fragments, request.historial)
    return {"answer": answer, "sources": sources}