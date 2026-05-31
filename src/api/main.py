from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.core.security import create_access_token, get_current_admin

from src.database.vector_store import VectorStoreManager
from src.database.db_manager import DBManager
from src.core.llm_engine import ChatEngine
from src.core.router import InputRouter
from src.core.retriever import AdvancedRetriever
from src.core.graph_retriever import GraphRetriever


# --------------------------
# App y middleware
# --------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)

app.mount("/pdfs",   StaticFiles(directory="data/raw"),      name="pdfs")
app.mount("/static", StaticFiles(directory="src/frontend"),  name="static")


# --------------------------
# Instancias globales
# --------------------------

vector_store       = VectorStoreManager(collection_name="etsi_hibrida")
chat_engine        = ChatEngine()
router             = InputRouter()
db_manager         = DBManager()
advanced_retriever = AdvancedRetriever(vector_store)
graph_retriever    = GraphRetriever()


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
    question:  str
    historial: list[HistorialMessage] = []

    @field_validator("question")
    @classmethod
    def question_not_empty_or_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La pregunta no puede estar vacía.")
        if len(v) > 1000:
            raise ValueError("La pregunta no puede superar los 1000 caracteres.")
        return v

class FeedbackRequest(BaseModel):
    log_id: int
    score:  int   # 1 = 👍, 0 = 👎

    @field_validator("score")
    @classmethod
    def score_valid(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("El score debe ser 0 o 1.")
        return v

class ConfigUpdateRequest(BaseModel):
    system_prompt: str | None = None
    model_name:    str | None = None


# --------------------------
# Endpoints
# --------------------------

@app.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Autenticar administrador y devolver JWT."""
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
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest):

    route = router.process_input(body.question)

    if not route.is_safe or not route.proceed_to_rag:
        log_id = db_manager.log_interaction(
            question=body.question,
            intent=route.intent.value,
            answer=route.direct_response,
            model_name="router",
            is_safe=route.is_safe,
            tokens_used=None,
            rerank_score=None,
        )
        return {"answer": route.direct_response, "sources": [], "log_id": log_id}

    historial_dicts = [m.model_dump() for m in body.historial]
    fragments, best_rerank_score = advanced_retriever.retrieve(body.question, historial_dicts)
    graph_context = graph_retriever.search(body.question, historial_dicts)

    if not fragments and not graph_context:
        log_id = db_manager.log_interaction(
            question=body.question,
            intent=route.intent.value,
            answer=NO_CONTEXT_MSG,
            model_name="none",
            is_safe=route.is_safe,
            tokens_used=None,
            rerank_score=None,
        )
        return {"answer": NO_CONTEXT_MSG, "sources": [], "log_id": log_id}

    seen: set[str] = set()
    sources = []
    if graph_context:
        sources.append({"title": "Grafo de Conocimiento ETSI", "url": None})
    for frag in fragments:
        meta = frag.get("metadata", {}) or {}

        source_url  = meta.get("source_url", "")
        source_file = meta.get("source_file", "")

        if not source_url:
            source_url = meta.get("url") or meta.get("source") or ""
        if not source_file:
            source_file = meta.get("source_path") or meta.get("source") or ""

        title = meta.get("title") or meta.get("Titulo") or ""

        dedup_key = source_url or source_file
        if not dedup_key or dedup_key in seen:
            continue
        seen.add(dedup_key)

        url = source_url if source_url else None
        sources.append({"title": title, "url": url})

    active_prompt     = db_manager.get_config("system_prompt") or None
    active_model_name = db_manager.get_config("model_name")    or None

    answer, tokens_used = chat_engine.generate_answer(
        body.question, fragments, body.historial,
        system_prompt=active_prompt,
        model_name=active_model_name,
        graph_context=graph_context,
    )

    log_id = db_manager.log_interaction(
        question=body.question,
        intent=route.intent.value,
        answer=answer,
        model_name=MODEL_NAME,
        is_safe=route.is_safe,
        tokens_used=tokens_used,
        rerank_score=best_rerank_score,
    )

    return {"answer": answer, "sources": sources, "log_id": log_id}


@app.post("/feedback")
@limiter.limit("60/minute")
async def feedback(request: Request, body: FeedbackRequest):
    ok = db_manager.update_feedback(body.log_id, body.score)
    if not ok:
        return {"ok": False, "error": "log_id no encontrado"}
    return {"ok": True}


@app.get("/admin/stats")
async def admin_stats(current_admin: str = Depends(get_current_admin)):
    return db_manager.get_admin_stats()


@app.get("/admin/config")
async def get_config(current_admin: str = Depends(get_current_admin)):
    """Devolver configuración activa del sistema."""
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
    """Actualizar prompt o modelo del LLM en caliente."""
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