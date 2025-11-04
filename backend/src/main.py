# backend/src/main.py

import logging
import asyncio
import aiofiles
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool # NEW IMPORT for blocking tasks

# --- CRITICAL FIX: Add the parent directory (backend/src) to sys.path ---
import sys 
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# --- Core Model Initialization ---
# NEW IMPORTS (You will need to ensure these modules are accessible)
from detection.detector import EnsembleDetector
#from reversion.reversion import FaceRestorer # Assuming this class exists

# --- Configuration ---
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}

# --- Global Models (Loaded once at startup) ---
detector = None
restorer = None

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="DeepSight v3.1 API (Sync Mode)", # UPDATED Title
    description="Optimized API for the Forensic Deepfake Reversion Pipeline (Direct Sync/Async).",
    version="3.1.0"
)


# --- Lifecycle Hooks ---
@app.on_event("startup")
async def startup_event():
    global detector, restorer
    TEMP_DIR.mkdir(exist_ok=True)
    
    # --- MODEL LOADING (Moved from worker.py) ---
    try:
        # NOTE: This model loading is synchronous and will delay startup.
        # Ensure the detector.py path fix is permanent!
        detector = EnsembleDetector(warmup=True)
        logger.info("✅ DeepSight API startup complete — Models loaded directly.")
    except Exception as e:
        logger.critical(f"❌ Fatal error loading models during startup: {e}")
        # Setting them to None ensures endpoints fail gracefully
        detector = None
        restorer = None


@app.on_event("shutdown")
async def shutdown_event():
    # Non-blocking cleanup
    for file in TEMP_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass
    logger.info("🧹 Temporary upload directory cleaned up.")


# --- Utility ---
async def save_temp_file(file: UploadFile) -> Path:
    # ... (Keep this function as is) ...
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_id = uuid.uuid4().hex
    # NOTE: Using original filename suffix is safer for libraries like PIL/timm
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    file_path = TEMP_DIR / f"{file_id}{suffix}"

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


# --- Synchronous Wrapper Functions ---
# These functions will run the blocking model inference code.
def sync_detect(file_path: Path) -> dict:
    if not detector:
        return {"error": "Detector model not loaded."}
    return detector.predict(str(file_path))


# NOTE: Assuming FaceRestorer.restore returns a path to the restored image.
def sync_revert(file_path: Path) -> dict:
    import base64 # Import needed only here for encoding
    if not restorer:
        return {"error": "Restorer model not loaded."}

    restored_image_path = None
    try:
        # 1. Perform restoration (blocking)
        # Assuming restorer.restore(input_path) returns the path to the output image
        restored_image_path = restorer.restore(str(file_path))

        # 2. Read and encode the result (blocking)
        with open(restored_image_path, "rb") as f:
            restored_image_bytes = f.read()
        
        encoded_image = base64.b64encode(restored_image_bytes).decode('utf-8')
        
        return {
            "message": "Reversion successful",
            "image_format": "png", # Assuming PNG output from restorer
            "restored_image_b64": encoded_image
        }
    except Exception as e:
        logger.error(f"Reversion failed during sync call: {e}")
        return {"error": str(e)}
    finally:
        # 3. Clean up the generated output file
        if restored_image_path and Path(restored_image_path).exists():
            Path(restored_image_path).unlink()


# --- Endpoints ---
@app.get("/", summary="Health check")
async def root():
    return {"status": "DeepSight API operational", "models_ready": detector is not None and restorer is not None}


@app.post("/detect", summary="Detect deepfakes synchronously")
async def detect_deepfake(file: UploadFile = File(...)):
    if not detector:
        raise HTTPException(status_code=503, detail="Service Unavailable: Detector model failed to load.")

    file_path = None
    try:
        # 1. Save file asynchronously
        file_path = await save_temp_file(file)

        # 2. Run blocking detection in a separate thread
        result = await run_in_threadpool(sync_detect, file_path)
        
        if "error" in result:
             raise Exception(result.get("error"))

        logger.info(f"🧠 Detection complete for {file_path.name}")
        return {"status": "SUCCESS", "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # 3. Clean up input file
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception:
            pass


@app.post("/revert", summary="Revert deepfake synchronously")
async def revert_image(file: UploadFile = File(...)):
    if not restorer:
        raise HTTPException(status_code=503, detail="Service Unavailable: Restorer model failed to load.")
    
    file_path = None
    try:
        # 1. Save file asynchronously
        file_path = await save_temp_file(file)

        # 2. Run blocking reversion in a separate thread
        result = await run_in_threadpool(sync_revert, file_path)

        if "error" in result:
             raise Exception(result.get("error"))

        logger.info(f"🎨 Reversion complete for {file_path.name}")
        return {"status": "SUCCESS", "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # 3. Clean up input file
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception:
            pass