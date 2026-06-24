"""F7 - Data export routes (async Download My Data)."""
import asyncio
import os
import tempfile
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from nl_to_sql.api.dependencies import get_current_user, get_session_service
from nl_to_sql.core.models.auth import UserPublic
from nl_to_sql.infrastructure.database.models import DataExportJob
from nl_to_sql.services.chat_session_service import ChatSessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/data", tags=["Data"])

# Temp directory where export ZIPs are stored until downloaded
_EXPORT_DIR = os.path.join(tempfile.gettempdir(), "nl2sql_exports")
os.makedirs(_EXPORT_DIR, mode=0o700, exist_ok=True)
try:
    os.chmod(_EXPORT_DIR, 0o700)
except OSError:
    pass


class ExportJobResponse(BaseModel):
    job_id: str
    status: str
    download_url: str | None = None
    error: str | None = None


@router.post("/export", status_code=202, summary="Request a full data export (async)")
async def request_export(
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> ExportJobResponse:
    """Queue a data export job. Poll GET /data/export/{job_id} for status."""
    async with session_service._session_factory() as db:
        job = DataExportJob(user_id=current_user.id, status="queued")
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_id = job.id

    # Launch background export
    _task = asyncio.create_task(_run_export(job_id, current_user.id, session_service._session_factory))
    _task.add_done_callback(lambda t: None)  # prevent GC
    logger.info("export job queued", job_id=job_id, user_id=current_user.id)

    return ExportJobResponse(job_id=job_id, status="queued")


@router.get("/export/{job_id}", response_model=ExportJobResponse, summary="Get export job status")
async def get_export_status(
    job_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> ExportJobResponse:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(DataExportJob).where(
                DataExportJob.id == job_id,
                DataExportJob.user_id == current_user.id,
            )
        )
        job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")

    download_url = None
    if job.status == "done" and job.artifact_path:
        download_url = f"/api/v1/data/download/{job_id}"

    return ExportJobResponse(
        job_id=job.id,
        status=job.status,
        download_url=download_url,
        error=job.error,
    )


@router.get("/download/{job_id}", summary="Download the completed data export ZIP")
async def download_export(
    job_id: str,
    current_user: UserPublic = Depends(get_current_user),
    session_service: ChatSessionService = Depends(get_session_service),
) -> FileResponse:
    async with session_service._session_factory() as db:
        result = await db.execute(
            select(DataExportJob).where(
                DataExportJob.id == job_id,
                DataExportJob.user_id == current_user.id,
            )
        )
        job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    if job.status != "done" or not job.artifact_path:
        raise HTTPException(status_code=400, detail="Export not ready yet")
    if not os.path.exists(job.artifact_path):
        raise HTTPException(status_code=410, detail="Export file has expired. Please request a new export.")

    from pathlib import Path
    export_dir = Path(_EXPORT_DIR).resolve()
    artifact = Path(job.artifact_path).resolve()
    if artifact.parent != export_dir:
        logger.error("Export path traversal detected", artifact_path=job.artifact_path, job_id=job_id)
        raise HTTPException(status_code=500, detail="Export file unavailable.")

    return FileResponse(
        path=job.artifact_path,
        media_type="application/zip",
        filename="my_data.zip",
    )


async def _run_export(job_id: str, user_id: str, session_factory: Any) -> None:
    """Background task to build user data ZIP."""
    import json
    import zipfile
    from datetime import datetime

    from sqlalchemy import select

    from nl_to_sql.infrastructure.database.models import (
        ChatMessage,
        ChatSession,
        DataExportJob,
        SavedQuery,
        UserInstructions,
        UserSettings,
    )

    async with session_factory() as db:
        result = await db.execute(select(DataExportJob).where(DataExportJob.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = "running"
            await db.commit()

    try:
        export_data: dict[str, Any] = {}
        async with session_factory() as db:
            r = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
            s = r.scalar_one_or_none()
            export_data["settings"] = {
                "sql_keyword_case": s.sql_keyword_case if s else "upper",
                "sql_indent": s.sql_indent if s else 2,
                "data_retention": s.data_retention if s else "forever",
            }

            r = await db.execute(select(UserInstructions).where(UserInstructions.user_id == user_id))
            instr = r.scalar_one_or_none()
            export_data["custom_instructions"] = {
                "content": instr.content if instr else "",
                "enabled": instr.enabled if instr else True,
            }

            r = await db.execute(select(SavedQuery).where(SavedQuery.user_id == user_id))
            saved = r.scalars().all()
            export_data["saved_queries"] = [
                {
                    "id": q.id,
                    "title": q.title,
                    "nl_prompt": q.nl_prompt,
                    "generated_sql": q.generated_sql,
                    "starred": q.starred,
                }
                for q in saved
            ]

            r = await db.execute(
                select(ChatSession.id, ChatSession.title, ChatMessage.timestamp,
                       ChatMessage.question, ChatMessage.sql, ChatMessage.dialect)
                .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.id, ChatMessage.timestamp)
            )
            export_data["query_history"] = [
                {
                    "session_id": row.id,
                    "session_title": row.title,
                    "timestamp": row.timestamp.isoformat(),
                    "question": row.question,
                    "sql": row.sql,
                    "dialect": row.dialect,
                }
                for row in r.all()
            ]

        zip_path = os.path.join(_EXPORT_DIR, f"{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.json", json.dumps(export_data, indent=2, default=str))
        try:
            os.chmod(zip_path, 0o600)
        except OSError:
            pass

        async with session_factory() as db:
            result = await db.execute(select(DataExportJob).where(DataExportJob.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "done"
                job.artifact_path = zip_path
                job.completed_at = datetime.utcnow()
                await db.commit()

    except Exception as exc:
        logger.error("export job failed", job_id=job_id, error=str(exc))
        async with session_factory() as db:
            result = await db.execute(select(DataExportJob).where(DataExportJob.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error = str(exc)
                await db.commit()
