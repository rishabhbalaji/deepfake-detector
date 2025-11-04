# backend/src/detection/detector.py

import torch
import timm
from PIL import Image
from pathlib import Path
import logging

# --- Configuration ---
# NOTE: Using the robust absolute path based on user input to avoid Celery environment issues
# The original logic (ROOT_DIR = Path(__file__).resolve().parents[2]) is complex and prone to errors.
# We are using the confirmed path for maximum stability in the worker environment.
MODEL_PATH = Path("/home/rbk/deepsight-forensics/backend/models/best_detector.pth")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Deepfake detector using a fine-tuned XceptionNet model.
    Optimized for inference performance and stability.
    """
    def __init__(self, warmup: bool = False):
        """
        Initialize model, transforms, and device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True  # Speed optimization for consistent input size
        torch.use_deterministic_algorithms(False)  # For performance; set True for strict reproducibility

        logger.info(f"Initializing EnsembleDetector on device: {self.device}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        # Model creation and loading
        self.model = timm.create_model("xception", pretrained=False, num_classes=1)
        
        # Load the state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        
        # --- FIX: Clean state dictionary keys for models saved with DataParallel/Compile wrappers ---
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Strip common prefixes like '_orig_mod.' or 'module.'
            if k.startswith('module.') or k.startswith('_orig_mod.'):
                cleaned_key = k.replace('_orig_mod.', '').replace('module.', '')
            else:
                cleaned_key = k
            cleaned_state_dict[cleaned_key] = v
        # -----------------------------------------------------------------------------------------

        self.model.load_state_dict(cleaned_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Transform setup
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        logger.info(f"Model '{MODEL_PATH.name}' loaded successfully.")

        if warmup and self.device == "cuda":
            self._warmup_model()

    def _warmup_model(self):
        """
        Runs one dummy inference to warm up CUDA kernels.
        Useful to remove first-inference latency.
        """
        dummy = torch.zeros((1, 3, 299, 299), device=self.device)  # Xception input size
        with torch.inference_mode():
            _ = self.model(dummy)
        torch.cuda.synchronize()
        logger.info("Warm-up completed for CUDA inference.")

    def predict(self, image_path: str) -> dict:
        """
        Predicts if an image is a deepfake.
        Returns dict: {"is_deepfake": bool, "confidence": float, "model_used": str}
        """
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transforms(img).unsqueeze(0).to(self.device, non_blocking=True)

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output = self.model(tensor)
                prob_real = torch.sigmoid(output).item()

            is_deepfake = prob_real < 0.5
            confidence = round(1 - prob_real if is_deepfake else prob_real, 4)

            return {
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_used": MODEL_PATH.name,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}