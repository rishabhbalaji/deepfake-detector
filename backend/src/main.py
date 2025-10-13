# backend/src/main.py

import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile

# Import our detector class
from .detection.detector import EnsembleDetector

# --- Application Setup ---
app = FastAPI(
    title="DeepSight v3.0 API",
    description="API for the Forensic Deepfake Reversion Pipeline.",
    version="1.0.0",
)

# --- Model Loading ---
detector = EnsembleDetector()

# --- Directory for Temporary Uploads ---
# Create a directory to store uploaded files temporarily
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "DeepSight Engine is online."}


@app.post("/detect/")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Receives an uploaded image file, saves it temporarily,
    and runs the deepfake detection model on it.
    """
    # Define the path where the file will be saved
    file_path = TEMP_UPLOAD_DIR / file.filename
    
    # Save the uploaded file to the temporary directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction using the path of the saved file
    prediction_result = detector.predict(image_path=str(file_path))
    
    # Add the file path to the result for reference
    prediction_result["file_processed"] = str(file_path)
    
    return prediction_result