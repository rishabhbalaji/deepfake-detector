# backend/src/reversion/reversion.py

import cv2
import torch
from pathlib import Path
import logging
import os

# Import the necessary GFP-GAN components
# This assumes you have cloned the GFPGAN repo and installed it
try:
    from gfpgan import GFPGANer
except ImportError:
    raise ImportError("GFPGAN not found. Please ensure it is cloned and installed correctly.")

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[3] # Navigate up to the project root
GFPGAN_DIR = ROOT_DIR / "GFPGAN"
MODEL_PATH = GFPGAN_DIR / "experiments/pretrained_models/GFPGANv1.4.pth"
OUTPUT_DIR = ROOT_DIR / "restored_images"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FaceRestorer:
    """
    A wrapper class for the GFPGAN face restoration model.
    """
    def __init__(self, upscale: int = 2):
        """
        Initializes the GFPGAN face restorer.

        Args:
            upscale (int): The upscaling factor for the model. Defaults to 2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing FaceRestorer on device: {self.device}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"GFP-GAN model not found at: {MODEL_PATH}. Please download pre-trained models.")

        # Set up the GFPGAN restorer
        self.gfpgan_enhancer = GFPGANer(
            model_path=str(MODEL_PATH),
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None, # We only focus on the face
            device=self.device
        )
        logger.info("FaceRestorer initialized successfully.")

    def restore(self, image_path: str) -> str:
        """
        Restores faces in a single image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            str: The file path of the restored image.
        """
        try:
            input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if input_img is None:
                raise ValueError(f"Failed to read image from path: {image_path}")

            # --- Run Restoration ---
            # The enhance method finds faces, crops them, restores them,
            # and pastes them back into the original image.
            _, _, restored_img = self.gfpgan_enhancer.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            # --- Save the Output ---
            img_name = Path(image_path).stem
            save_path = OUTPUT_DIR / f"{img_name}_restored.png"
            
            cv2.imwrite(str(save_path), restored_img)
            logger.info(f"Restored image saved to: {save_path}")

            return str(save_path)

        except Exception as e:
            logger.error(f"Error during face restoration: {e}")
            raise  # Re-raise the exception to be handled by the API