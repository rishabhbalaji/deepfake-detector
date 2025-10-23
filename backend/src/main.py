# backend/src/main.py

import logging
import asyncio
import aiofiles
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from worker import celery_app, run_detection_task, run_reversion_task

# --- Configuration ---
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="DeepSight v3.1 API",
    description="Optimized API for the Forensic Deepfake Reversion Pipeline (Async + Celery).",
    version="3.1.0"
)


# --- Lifecycle Hooks ---
@app.on_event("startup")
async def startup_event():
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info("✅ DeepSight API startup complete — Celery workers will handle model inference.")


@app.on_event("shutdown")
async def shutdown_event():
    # Non-blocking cleanup to avoid interfering with running Celery tasks
    for file in TEMP_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass
    logger.info("🧹 Temporary upload directory cleaned up.")


# --- Utility ---
async def save_temp_file(file: UploadFile) -> Path:
    """Save uploaded file asynchronously with validation and limits."""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_id = uuid.uuid4().hex
    file_path = TEMP_DIR / f"{file_id}_{file.filename}"

    size = 0
    async with aiofiles.open(file_path, "wb") as buffer:
        while chunk := await file.read(8192):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                await buffer.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (limit 10MB).")
            await buffer.write(chunk)

    return file_path


# --- Endpoints ---
@app.get("/", summary="Health check")
async def root():
    return {"status": "DeepSight API operational"}


@app.post("/detect", summary="Detect deepfakes asynchronously")
async def detect_deepfake(file: UploadFile = File(...)):
    try:
        file_path = await save_temp_file(file)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        task = run_detection_task.delay(file_bytes)
        logger.info(f"🧠 Detection task queued: {task.id}")
        return JSONResponse(status_code=202, content={"task_id": task.id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


@app.post("/revert", summary="Revert deepfake asynchronously")
async def revert_image(file: UploadFile = File(...)):
    try:
        file_path = await save_temp_file(file)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        task = run_reversion_task.delay(file_bytes)
        logger.info(f"🎨 Reversion task queued: {task.id}")
        return JSONResponse(status_code=202, content={"task_id": task.id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reversion failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


@app.get("/results/{task_id}", summary="Retrieve task results")
async def get_task_result(task_id: str):
    try:
        task_result = AsyncResult(task_id, app=celery_app)

        if not task_result.ready():
            return {"status": "PENDING"}

        if task_result.successful():
            return {"status": "SUCCESS", "result": task_result.result}

        error_msg = str(task_result.info)
        logger.warning(f"⚠️ Task {task_id} failed: {error_msg}")
        return JSONResponse(status_code=500, content={"status": "FAILURE", "error": error_msg})

    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch task result.")
