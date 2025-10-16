# backend/src/main.py

import asyncio
import aiofiles
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .detection.detector import EnsembleDetector

# --- Configuration ---
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}

# --- Application Setup ---
app = FastAPI(
    title="DeepSight v3.0 API",
    description="Optimized API for the Forensic Deepfake Reversion Pipeline.",
    version="1.0.0",
)

# --- Model Loading (global singleton) ---
detector = None
_model_lock = asyncio.Lock()


@app.on_event("startup")
async def load_model():
    """
    Load the model once on startup (non-blocking).
    Uses a lock to prevent concurrent inits under high load.
    """
    global detector
    async with _model_lock:
        if detector is None:
            loop = asyncio.get_event_loop()
            detector = await loop.run_in_executor(None, EnsembleDetector, False)
            print("✅ DeepSight model initialized and ready.")


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {"status": "DeepSight Engine is online."}


@app.post("/detect/")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Async endpoint to process uploaded image and run inference.
    """
    # --- Validation ---
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    if detector is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Try again shortly.")

    # --- Generate safe temp file path ---
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = TEMP_UPLOAD_DIR / unique_name

    try:
        # --- Save file asynchronously ---
        async with aiofiles.open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                await buffer.write(chunk)

        # --- Run inference in background thread ---
        loop = asyncio.get_event_loop()
        prediction_result = await loop.run_in_executor(
            None, detector.predict, str(file_path)
        )

        # --- Add reference to file processed ---
        prediction_result["file_processed"] = str(file_path)

        return JSONResponse(content=prediction_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    finally:
        # --- Cleanup ---
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
