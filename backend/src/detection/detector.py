# src/detection/detector.py

import torch
import timm
from PIL import Image

class EnsembleDetector:
    """
    A class to encapsulate the multi-model deepfake detection logic.
    This will manage loading the models and running inference.
    """
    def __init__(self):
        """
        Initializes the detector and loads the XceptionNet model.
        """
        # Set the device (use GPU if available, otherwise CPU)
        self.device = "cpu"
        print(f"EnsembleDetector initialized. Using device: {self.device}")

        # Load pre-trained XceptionNet model using timm
        # We replace the final layer for binary classification (real vs. fake)
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)
        
        # Move the model to the selected device
        self.model.to(self.device)
        
        # Set the model to evaluation mode (important for inference)
        self.model.eval()

        # Get the model-specific transformations
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def predict(self, image_path: str) -> dict:
        """
        Predicts if an image is a deepfake using the loaded model.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            dict: A dictionary containing the prediction and confidence score.
        """
        try:
            # Open the image file and convert to RGB
            img = Image.open(image_path).convert("RGB")
            
            # Apply the transformations and add a batch dimension
            tensor = self.transforms(img).unsqueeze(0).to(self.device)

            # Perform inference without calculating gradients
            with torch.no_grad():
                output = self.model(tensor)
                # Apply sigmoid to get a probability score between 0 and 1
                probability = torch.sigmoid(output).item()

            # Determine prediction based on a 0.5 threshold
            is_deepfake = probability > 0.5
            confidence = probability if is_deepfake else 1 - probability

            return {
                "is_deepfake": is_deepfake,
                "confidence": round(confidence, 4),
                "models_consulted": ["XceptionNet"]
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}