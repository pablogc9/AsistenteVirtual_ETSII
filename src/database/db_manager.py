from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Text, create_engine, func, select, text as sql_text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

#--------------------------
# Modelo ORM
#--------------------------

class Base(DeclarativeBase):
    pass

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id:             Mapped[int]                 = mapped_column(primary_key=True, autoincrement=True)
    timestamp:      Mapped[datetime.datetime]   = mapped_column(default=datetime.datetime.now)
    question:       Mapped[str]
    intent:         Mapped[str]
    answer:         Mapped[str]
    tokens_used:    Mapped[Optional[int]]       = mapped_column(nullable=True)
    model_name:     Mapped[str]               
    feedback:       Mapped[Optional[int]]       = mapped_column(nullable=True)     # 1 = 👍,  0 = 👎
    is_safe:        Mapped[bool]
    rerank_score:   Mapped[Optional[float]]     = mapped_column(nullable=True)     # CrossEncoder score del mejor fragmento

class SystemConfig(Base):
    """
    Tabla clave-valor para configuración dinámica del sistema
    (system prompt, nombre del modelo, etc.).
    Si una clave no existe, los métodos del manager devuelven el valor por defecto.
    """
    __tablename__ = "system_config"

    key:   Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column(Text)


#--------------------------
# Manager
#--------------------------

class DBManager:
    """Gestión de logs de interacción y configuración dinámica en SQLite."""

    def __init__(self, db_path: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_path = project_root / "data" / "chatlogs.db"
        default_path.parent.mkdir(parents=True, exist_ok=True)

        path = db_path or str(default_path)
        self._engine = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(self._engine)

        # Migración: añade columna rerank_score si la DB ya existía sin ella
        try:
            with self._engine.connect() as conn:
                conn.execute(sql_text(
                    "ALTER TABLE chat_logs ADD COLUMN rerank_score REAL"
                ))
                conn.commit()
        except Exception:
            pass

    def log_interaction(
        self,
        question:     str,
        intent:       str,
        answer:       str,
        model_name:   str,
        is_safe:      bool,
        tokens_used:  int | None   = None,
        rerank_score: float | None = None,
    ) -> int:
        """Registrar interacción y devolver el ID del log."""
        with Session(self._engine) as session:
            log = ChatLog(
                question=question,
                intent=intent,
                answer=answer,
                tokens_used=tokens_used,
                model_name=model_name,
                is_safe=is_safe,
                rerank_score=rerank_score,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log.id

    def update_feedback(self, log_id: int, score: int) -> bool:
        """Actualizar feedback (0/1) de un registro existente."""
        if score not in (0, 1):
            raise ValueError("score debe ser 0 (negativo) o 1 (positivo).")

        with Session(self._engine) as session:
            log = session.get(ChatLog, log_id)
            if log is None:
                return False
            log.feedback = score
            session.commit()
            return True

    def get_admin_stats(self) -> dict[str, Any]:
        """Devolver métricas agregadas para el panel de administración."""
        with Session(self._engine) as session:
            total = session.scalar(func.count(ChatLog.id)) or 0
            total_tokens = session.scalar(func.sum(ChatLog.tokens_used)) or 0

            intent_rows = session.execute(
                select(ChatLog.intent, func.count(ChatLog.id))
                .group_by(ChatLog.intent)
            ).all()
            intent_distribution = {row[0]: row[1] for row in intent_rows}

            avg_feedback = session.scalar(
                select(func.avg(ChatLog.feedback))
                .where(ChatLog.feedback.is_not(None))
            )

            safe_count = session.scalar(
                select(func.count(ChatLog.id))
                .where(ChatLog.is_safe == True)
            ) or 0

        return {
            "total_interactions":   total,
            "total_tokens":         total_tokens,
            "intent_distribution":  intent_distribution,
            "avg_feedback":         float(avg_feedback) if avg_feedback is not None else None,
            "safe_percentage":      round(safe_count / total * 100, 1) if total else 0.0,
        }

    def get_config(self, key: str, default: str = "") -> str:
        """Recuperar valor de configuración por clave."""
        with Session(self._engine) as session:
            row = session.get(SystemConfig, key)
            return row.value if row is not None else default

    def set_config(self, key: str, value: str) -> None:
        """Insertar o actualizar entrada de configuración."""
        with Session(self._engine) as session:
            row = session.get(SystemConfig, key)
            if row is None:
                session.add(SystemConfig(key=key, value=value))
            else:
                row.value = value
            session.commit()

    def get_all_config(self) -> dict[str, str]:
        """Devuelve todas las claves de configuración como diccionario."""
        with Session(self._engine) as session:
            rows = session.execute(select(SystemConfig)).scalars().all()
            return {row.key: row.value for row in rows}

    def get_recent_logs(
        self,
        page:      int = 1,
        page_size: int = 10,
        intent:    str | None = None,
        feedback:  str | None = None,   # "1" | "0" | "none" | None
        is_safe:   bool | None = None,
    ) -> dict[str, Any]:
        """Devolver logs paginados con filtros opcionales."""
        # ── Construcción dinámica de filtros ──────────────────────────────────
        filters = []
        if intent:
            filters.append(ChatLog.intent == intent)
        if feedback == "1":
            filters.append(ChatLog.feedback == 1)
        elif feedback == "0":
            filters.append(ChatLog.feedback == 0)
        elif feedback == "none":
            filters.append(ChatLog.feedback.is_(None))
        if is_safe is not None:
            filters.append(ChatLog.is_safe == is_safe)

        with Session(self._engine) as session:
            # ── Total de registros que cumplen los filtros ────────────────────
            count_q = select(func.count(ChatLog.id))
            data_q  = select(ChatLog)
            if filters:
                count_q = count_q.where(*filters)
                data_q  = data_q.where(*filters)

            total = session.scalar(count_q) or 0

            # ── Página de datos ───────────────────────────────────────────────
            offset = (page - 1) * page_size
            rows = session.execute(
                data_q.order_by(ChatLog.timestamp.desc())
                       .offset(offset)
                       .limit(page_size)
            ).scalars().all()

            items = [
                {
                    "id":          row.id,
                    "timestamp":   row.timestamp.isoformat(),
                    "question":    row.question,
                    "intent":      row.intent,
                    "answer":      row.answer,
                    "tokens_used": row.tokens_used,
                    "model_name":  row.model_name,
                    "feedback":      int(row.feedback) if row.feedback is not None else None,
                    "is_safe":       row.is_safe,
                    "rerank_score":  round(row.rerank_score, 3) if row.rerank_score is not None else None,
                }
                for row in rows
            ]

        return {
            "items":       items,
            "total":       total,
            "page":        page,
            "page_size":   page_size,
            "total_pages": max(1, math.ceil(total / page_size)),
        }


__all__ = ["DBManager", "ChatLog", "SystemConfig"]