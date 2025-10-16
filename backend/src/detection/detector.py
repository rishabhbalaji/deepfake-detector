# src/detection/detector.py
import torch
import timm
from PIL import Image
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[2] # Navigate up to the project root
MODEL_PATH = ROOT_DIR / "models/best_detector.pth"

class EnsembleDetector:
    """
    A class to encapsulate the deepfake detection logic.
    It now loads our custom-trained XceptionNet model.
    """
    def __init__(self):
        """
        Initializes the detector and loads the fine-tuned model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"EnsembleDetector initialized. Using device: {self.device}")

        # 1. Create the model architecture (without pre-trained weights)
        self.model = timm.create_model('xception', pretrained=False, num_classes=1)
        
        # 2. Load our custom-trained weights from the .pth file
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        
        # 3. Prepare the model for inference
        self.model.to(self.device)
        self.model.eval()

        # 4. Get the model-specific transformations
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def predict(self, image_path: str) -> dict:
        """
        Predicts if an image is a deepfake using our trained model.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transforms(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probability = torch.sigmoid(output).item()

            is_deepfake = probability > 0.5
            confidence = probability if is_deepfake else 1 - probability

            return {
                "is_deepfake": is_deepfake,
                "confidence": round(confidence, 4),
                "model_used": str(MODEL_PATH.name)
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}