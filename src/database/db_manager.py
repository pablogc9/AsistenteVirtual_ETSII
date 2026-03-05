from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

#--------------------------
# Modelo ORM
#--------------------------

class Base(DeclarativeBase):
    pass

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id:             Mapped[int]                 = mapped_column(primary_key=True, autoincrement=True)
    timestamp:      Mapped[datetime.datetime]   = mapped_column(default=datetime.datetime.utcnow)
    question:       Mapped[str]
    intent:         Mapped[str]
    answer:         Mapped[str]
    tokens_used:    Mapped[Optional[int]]       = mapped_column(nullable=True)
    model_name:     Mapped[str]               
    feedback:       Mapped[Optional[str]]       = mapped_column(nullable=True)     # 1 = 👍,  0 = 👎
    is_safe:        Mapped[bool]

#--------------------------
# Manager
#--------------------------

class DBManager:
    """
    Gestiona la base de datos SQLite de logs de interacciones.

    La base de datos se crea automáticamente en data/chatlogs.db
    la primera vez que se instancia la clase.
    """

    def __init__(self, db_path: str | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_path = project_root / "data" / "chatlogs.db"
        default_path.parent.mkdir(parents=True, exist_ok=True)

        path = db_path or str(default_path)
        self._engine = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(self._engine)

    def log_interaction(
        self,
        question:    str,
        intent:      str,
        answer:      str,
        model_name:  str,
        is_safe:     bool,
        tokens_used: int | None = None,
    ) -> int:
        """
        Registra una interacción completa y devuelve el ID del registro.
        (necesario para poder actualizar el feedback después)
        """
        with Session(self._engine) as session:
            log = ChatLog(
                question=question,
                intent=intent,
                answer=answer,
                tokens_used=tokens_used,
                model_name=model_name,
                is_safe=is_safe,
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log.id

    def update_feedback(self, log_id: int, score: int) -> bool:
        """
        Actualiza el feedback (1 = 👍, 0 = 👎) de un registro existente.
        Devuelve False si el log_id no existe.
        """
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
        """
        Devuelve métricas agregadas para el panel de control:
        - total de interacciones
        - total de tokens consumidos
        - distribución de intenciones (para gráfica de tarta)
        - media de feedback (para gráfica de satisfacción)
        - porcentaje de interacciones seguras
        """
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
            "avg_feedback":         round(avg_feedback, 2) if avg_feedback is not None else None,
            "safe_percentage":      round(safe_count / total * 100, 1) if total else 0.0,
        }

__all__ = ["DBManager", "ChatLog"]