# DeepSight Project: Research and Methodology Log

This document tracks the key decisions, procedures, and results throughout the Forensic Deepfake Reversion Pipeline project.

## Phase 1: Detection Module

### 1.1: Environment Setup
- **OS:** Arch Linux via WSL 2
- **Python:** Miniconda environment 'deepsight'
- **Key Libraries:** PyTorch, FastAPI, Timm, OpenCV

### 1.2: Data Acquisition & Preparation
- **Dataset:** FaceForensics++ (FF++)
- **Reasoning:** Selected as a standard academic benchmark for comparable results.
- **Data Subset:** Used `original_sequences` and `manipulated_sequences/Deepfakes`.
- **Compression:** Chose `c23` to balance visual fidelity with manageable file size.
- **Frame Extraction:** Extracted **20 evenly-spaced frames** per video, resulting in a dataset of **40,000 images** (32,000 for training, 8,000 for validation).

### 1.3: Model Fine-Tuning (Current Stage)
- **Base Model:** XceptionNet (pre-trained on ImageNet).
- **Reasoning:** A powerful CNN architecture proven to be effective for image classification tasks, serving as a strong baseline.
