# backend/src/worker.py

import logging
# The original CRITICAL FIX is still required for the relative imports to work
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------------------------

# Import our project-specific modules - Now guaranteed to work
from detection.detector import EnsembleDetector
from reversion.reversion import FaceRestorer

# --- Celery Configuration ---
# REMOVED CELERY APP, BROKER, AND TASKS

# Create stubs for main.py to import without error
class CeleryAppStub:
    pass
celery_app = CeleryAppStub()

# Stubs for tasks (no longer used, but kept for minimal changes elsewhere)
def run_detection_task(image_data: bytes):
    raise NotImplementedError("Celery tasks are disabled in sync mode.")
    
def run_reversion_task(image_data: bytes):
    raise NotImplementedError("Celery tasks are disabled in sync mode.")