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

---
### Log Entry: October 16, 2025

**Analysis of Initial Training Runs (V1, V2, V3):**

Completed three major training experiments to establish a baseline for the XceptionNet detector.

-   **Run 1 (V1):** Basic training, 50 epochs. Best Val Loss: `0.2159`.
-   **Run 2 (V2):** Optimized with AMP/`torch.compile`, 100 epochs. Best Val Loss: `0.2000`.
-   **Run 3 (V3):** Extended run, 200 epochs, 16 workers. Best Val Loss: `0.2078`.

**Conclusion:** The model from **Run 2 (`best_detector_v2_88acc.pth`) is the current champion**, achieving the lowest validation loss. The extended 200-epoch run confirmed that peak performance is reached relatively early and becomes unstable in later epochs.

**Next Step: Advanced Fine-Tuning Run (V4):**

Proceeding with a final, advanced fine-tuning experiment (`train_finetune.py`). The objective is to determine if a more sophisticated training strategy can surpass the `0.2000` validation loss benchmark.

-   **Key Techniques:** Label Smoothing, Cosine Annealing LR, Gradual Unfreezing, Early Stopping (`patience=15`).
-   **Expected Outcome:** A potentially lower validation loss, representing the performance ceiling for this architecture.
-   
---
---
### Log Entry: November 4, 2025 (CRITICAL UPDATE)

**Detection Module Final Model Selection (Phase 1 Complete)**

A breakthrough result was achieved using the `train_master_v8.py` configuration (1e-3 LR + Cosine Annealing LR). The previous champion validation loss of 0.2000 was dramatically surpassed.

-   **Run V8 (Final):** Base XceptionNet, 1e-3 LR, CosineAnnealingLR.
-   **Best Val Loss:** **0.0676**
-   **Model File:** `detector_xception_final_0.0676.pth`

**Final Model Selection:**
The model `detector_xception_final_0.0676.pth` is now the **definitive champion** for the Detection Module. This result is confirmed to be free of data leakage.

**Phase 1 Completion:**
The Detection Module R&D phase is complete. The next immediate step is to update the detector implementation to load this new model and formally proceed to Phase 2: GAN-Based Reversion and Confidence Scoring.