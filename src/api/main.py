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
from src.core.retriever import AdvancedRetriever


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

vector_store       = VectorStoreManager()
chat_engine        = ChatEngine()
router             = InputRouter()
db_manager         = DBManager()
advanced_retriever = AdvancedRetriever(vector_store)


# --------------------------
# Constantes
# --------------------------

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

class ConfigUpdateRequest(BaseModel):
    system_prompt: str | None = None
    model_name:    str | None = None


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
            rerank_score=None,
        )
        return {"answer": route.direct_response, "sources": [], "log_id": log_id}

    # --- Capa de RAG con Multi-Query + Re-ranking ---
    fragments, best_rerank_score = advanced_retriever.retrieve(request.question)

    if not fragments:
        log_id = db_manager.log_interaction(
            question=request.question,
            intent=route.intent.value,
            answer=NO_CONTEXT_MSG,
            model_name="none",
            is_safe=route.is_safe,
            tokens_used=None,
            rerank_score=None,
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
    # Leer config dinámica; si la clave no existe en DB usa los valores por defecto
    active_prompt     = db_manager.get_config("system_prompt") or None
    active_model_name = db_manager.get_config("model_name")    or None

    answer, tokens_used = chat_engine.generate_answer(
        request.question, fragments, request.historial,
        system_prompt=active_prompt,
        model_name=active_model_name,
    )

    log_id = db_manager.log_interaction(
        question=request.question,
        intent=route.intent.value,
        answer=answer,
        model_name=MODEL_NAME,
        is_safe=route.is_safe,
        tokens_used=tokens_used,
        rerank_score=best_rerank_score,
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


@app.get("/admin/config")
async def get_config(current_admin: str = Depends(get_current_admin)):
    """
    Devuelve la configuración activa del sistema.
    Si una clave no se ha editado todavía, indica que se está usando el valor por defecto.
    """
    from src.core.llm_engine import SYSTEM_PROMPT
    stored = db_manager.get_all_config()
    return {
        "system_prompt": stored.get("system_prompt", SYSTEM_PROMPT),
        "model_name":    stored.get("model_name",    MODEL_NAME),
    }


@app.put("/admin/config")
async def update_config(
    request:       ConfigUpdateRequest,
    current_admin: str = Depends(get_current_admin),
):
    """
    Actualiza la configuración del sistema en caliente (sin reiniciar el servidor).
    Solo se guardan los campos que vengan con valor en el body.
    """
    if request.system_prompt is not None:
        db_manager.set_config("system_prompt", request.system_prompt)
    if request.model_name is not None:
        db_manager.set_config("model_name", request.model_name)
    return {"ok": True, "updated": db_manager.get_all_config()}


@app.get("/admin/logs")
async def get_logs(
    page:          int        = 1,
    page_size:     int        = 10,
    intent:        str | None = None,
    feedback:      str | None = None,   # "1" | "0" | "none"
    is_safe:       bool | None = None,
    current_admin: str        = Depends(get_current_admin),
):
    return db_manager.get_recent_logs(
        page=page,
        page_size=page_size,
        intent=intent or None,
        feedback=feedback or None,
        is_safe=is_safe,
    )