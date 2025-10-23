# backend/src/worker.py

import logging
from pathlib import Path
import tempfile
import base64

from celery import Celery

# Import our project-specific modules
from detection.detector import DeepSightDetector
from reversion.reversion import FaceRestorer

# --- Celery Configuration ---
# This sets up Celery to use your local Redis server as the message broker and result backend.
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Model Pre-loading ---
# These models will be loaded *once* per worker process when it starts,
# which is highly efficient.
try:
    detector = DeepSightDetector(warmup=True)
    restorer = FaceRestorer()
    logger.info("Models loaded successfully in Celery worker.")
except Exception as e:
    logger.critical(f"Fatal error loading models in worker: {e}")
    detector = None
    restorer = None

# --- Celery Tasks ---

@celery_app.task(name="detection_task")
def run_detection_task(image_data: bytes) -> dict:
    """Celery task to run deepfake detection in the background."""
    if not detector:
        return {"error": "Detector model not loaded in worker."}
    
    # Use a temporary file to pass the image data to the detector
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name

    try:
        result = detector.predict(tmp_path)
        return result
    except Exception as e:
        logger.error(f"Detection task failed: {e}")
        return {"error": str(e)}
    finally:
        Path(tmp_path).unlink() # Clean up the temp file

@celery_app.task(name="reversion_task")
def run_reversion_task(image_data: bytes) -> dict:
    """Celery task to run face reversion/restoration in the background."""
    if not restorer:
        return {"error": "Restorer model not loaded in worker."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_input:
        tmp_input.write(image_data)
        input_path = tmp_input.name
    
    restored_image_path = None # Initialize to handle potential errors
    try:
        # Perform the restoration, which saves a new file
        restored_image_path = restorer.restore(input_path)

        # Read the restored image and encode it as a base64 string
        with open(restored_image_path, "rb") as f:
            restored_image_bytes = f.read()
        
        encoded_image = base64.b64encode(restored_image_bytes).decode('utf-8')
        
        return {
            "message": "Reversion successful",
            "image_format": "png",
            "restored_image_b64": encoded_image
        }
    except Exception as e:
        logger.error(f"Reversion task failed: {e}")
        return {"error": str(e)}
    finally:
        # Clean up both the temporary input and the generated output file
        Path(input_path).unlink()
        if restored_image_path and Path(restored_image_path).exists():
            Path(restored_image_path).unlink()