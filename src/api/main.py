from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from src.core.security import create_access_token, get_current_admin
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.database.vector_store import VectorStoreManager
from src.database.db_manager import DBManager
from src.core.llm_engine import ChatEngine
from src.core.router import InputRouter


# --------------------------
# App y middleware
# --------------------------

app = FastAPI()

# Middleware CORS: debe registrarse antes de definir cualquier ruta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Acepta peticiones desde cualquier dominio
    allow_methods=["*"],      # Acepta GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],      # Acepta cualquier cabecera HTTP
)

app.mount("/pdfs", StaticFiles(directory="data/raw"), name="pdfs")


# --------------------------
# Instancias globales
# --------------------------

vector_store = VectorStoreManager()
chat_engine = ChatEngine()
router = InputRouter()
db_manager = DBManager()


# --------------------------
# Constantes
# --------------------------

DISTANCE_THRESHOLD  = 0.6
NO_CONTEXT_MSG      = "Lo siento, no he encontrado información oficial sobre esa pregunta en los documentos de la ETSI Informática."
BASE_URL            = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME          = "llama-3.1-8b-instant"


# --------------------------
# Esquemas Pydantic
# --------------------------

class HistorialMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    historial: list[HistorialMessage] = []

class FeedbackRequest(BaseModel):
    log_id: int
    score:  int   # 1 = 👍, 0 = 👎


# --------------------------
# Endpoints
# --------------------------

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Recibe username y password y verifica contra las variables de entorno ADMIN_USERNAME y ADMIN_PASSWORD
    Si son correctas, devuelve un JWT válido durante 8 horas
    """
    admin_user = os.getenv("ADMIN_USERNAME", "")
    admin_pass = os.getenv("ADMIN_PASSWORD", "")

    if not admin_user or not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Credenciales de administrador no configuradas en el servidor."
        )

    if form_data.username != admin_user or form_data.password != admin_pass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/ask")
async def ask(request: AskRequest):

    # --- Capa de enrutamiento y seguridad ---
    route = router.process_input(request.question)

    if not route.is_safe or not route.proceed_to_rag:  # Bloqueo por seguridad o charla casual: Respuesta directa
        log_id = db_manager.log_interaction(
            question=request.question,
            intent=route.intent.value,
            answer=route.direct_response,
            model_name="router",
            is_safe=route.is_safe,
            tokens_used=None,
        )
        return {"answer": route.direct_response, "sources": [], "log_id": log_id}

    # --- Capa de RAG ---
    fragments = vector_store.search(request.question, k=5)

    # Si todos los fragmentos superan el umbral de distancia, la pregunta
    # está demasiado lejos del contenido indexado y no respondemos
    if not fragments or all(frag["distance"] > DISTANCE_THRESHOLD for frag in fragments):
        log_id = db_manager.log_interaction(
            question=request.question,
            intent=route.intent.value,
            answer=NO_CONTEXT_MSG,
            model_name="none",
            is_safe=route.is_safe,
            tokens_used=None,
        )
        return {"answer": NO_CONTEXT_MSG, "sources": [], "log_id": log_id}

    # Construcción de fuentes únicas con título y URL/ruta
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
        url = f"{BASE_URL}/pdfs/{Path(raw_source).name}" if raw_source.endswith(".pdf") else raw_source
        sources.append({"title": title, "url": url})

    # -- Capa de LLM ---
    answer, tokens_used = chat_engine.generate_answer(
        request.question, fragments, request.historial
    )

    log_id = db_manager.log_interaction(
        question=request.question,
        intent=route.intent.value,
        answer=answer,
        model_name=MODEL_NAME,
        is_safe=route.is_safe,
        tokens_used=tokens_used,
    )

    return {"answer": answer, "sources": sources, "log_id": log_id}


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    ok = db_manager.update_feedback(request.log_id, request.score)
    if not ok:
        return {"ok": False, "error": "log_id no encontrado"}
    return {"ok": True}


@app.get("/admin/stats")
async def admin_stats(current_admin: str = Depends(get_current_admin)):
    return db_manager.get_admin_stats()