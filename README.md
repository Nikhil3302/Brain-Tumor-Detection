# An Automated Deep Learning Framework for Brain Tumor Detection and Classification Using MRI

## 1. Abstract & Overview
Brain tumor detection and classification from Magnetic Resonance Imaging (MRI) scans is a crucial task in clinical diagnosis. Manual analysis of MRI scans is time-consuming and prone to human variability.

To address these challenges, this project implements the hybrid automated deep learning framework proposed in our research paper:
1. **Localization**: Single-stage tumor localization using **YOLO-NAS** (Neural Architecture Search).
2. **Region of Interest (ROI) Extraction**: Automated extraction of neoplastic tissue regions, eliminating background noise and manual ROI selection.
3. **Fine-grained Classification**: **EfficientNet** classifies extracted ROIs into **Glioma**, **Meningioma**, **Pituitary Tumor**, or **Non-Tumor**.
4. **CAD System Deployment**: Interactive Computer-Aided Diagnosis (CAD) Web Workstation for radiologists and clinicians.

---

## 2. Methodology & Architecture

```
+-----------------------------------------------------------------------------------+
|                                 MRI Acquisition                                   |
|                          (BraTS, REMBRANDT, BR35H)                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              Image Preprocessing                                  |
|            - Noise Removal (Median/Gaussian)   - Intensity Normalization [0, 1]   |
|            - Contrast Enhancement (CLAHE)      - Resizing & Multi-axis Alignment  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Stage 1: YOLO-NAS Tumor Detection                          |
|             - Multi-Scale Neural Architecture Search Backbone (QARepVGG)          |
|             - Path Aggregation Network (PAN Neck)                                 |
|             - Decoupled Bounding Box Regression & Tumor Objectness Head           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     Stage 2: Region of Interest (ROI) Extraction                  |
|                   - Contextual Cropping with Dynamic Boundary Margin              |
|                   - Noise Suppression & Standardized 224x224 Rescaling            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       Stage 3: EfficientNet ROI Classification                    |
|                - Compound Scaling (Depth, Width, Resolution)                      |
|                - Multi-Class Logits: Glioma | Meningioma | Pituitary | Non-Tumor   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               Diagnostic CAD Output                               |
|        - Localized Tumor Bounding Box Overlay  - Predicted Pathology & Probability|
|        - Binary Malignancy Confidence          - Printable Diagnostic Report      |
+-----------------------------------------------------------------------------------+
```

---

## 3. Comparative Results (Paper Benchmarks)

### Table 1: Performance Benchmark of Algorithms
| Metric / Parameter | CNN | VGG16 | Hybrid Approach (YOLO-NAS + EfficientNet) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 83% | 85% | **88%** *(+5.0% gain)* |
| **Balanced Accuracy** | 79% | 82% | **86%** *(+7.0% gain)* |
| **Sensitivity (Recall)** | 78% | 81% | **85%** *(+7.0% gain)* |
| **Specificity** | 84% | 86% | **89%** *(+5.0% gain)* |

### Figure 6: Confusion Matrix Validation (N = 1,000 cases)
* **True Non-Tumor Correctly Identified (TN)**: 485
* **False Positives (FP)**: 15
* **False Negatives (FN)**: 10
* **True Tumors Correctly Identified (TP)**: 490
* **Derived Overall Test Accuracy**: **97.5%**
* **Sensitivity / True Tumor Recall**: **98.0%**
* **Specificity**: **97.0%**

---

## 4. Project Directory Structure
```
Brain Tumor Detection/
├── .venv/                         # Dedicated Python 3.12 Virtual Environment
├── app/                           # Interactive CAD Web Application
│   ├── app.py                     # Flask backend & diagnosis API routes
│   ├── static/                    # Styling and scripts
│   └── templates/                 # Modern medical dashboard template (index.html)
├── dataset/                       # Dataset handling & sample scans
│   └── samples/                   # Curated MRI scans (Glioma, Meningioma, Pituitary, Non-Tumor)
├── models_saved/                  # Trained neural network weights (.pth checkpoints)
│   ├── efficientnet_b0_best.pth   # Fine-tuned EfficientNet classifier
│   └── yolo_nas_best.pth          # Fine-tuned YOLO-NAS detector
├── outputs/                       # Generated figures, confusion matrix & reports
│   ├── figure_6_confusion_matrix.png
│   ├── figure_7_comparative_graph.png
│   └── pipeline_test_*.png        # Visual 4-panel diagnostic reports
├── src/                           # Core Framework Modules
│   ├── __init__.py
│   ├── baselines.py               # Standard CNN and VGG16 implementations
│   ├── classifier_efficientnet.py # EfficientNet architecture & ROI classifier
│   ├── dataset.py                 # MRI dataset loader & synthetic anatomical generator
│   ├── detector_yolo_nas.py       # YOLO-NAS architecture with QARepVGG & PAN neck
│   ├── metrics.py                 # Diagnostic metrics & publication plotting (Fig 6 & 7)
│   ├── pipeline.py                # End-to-end CAD pipeline coordinator
│   ├── preprocessing.py           # Noise removal, CLAHE enhancement & normalization
│   ├── roi_extractor.py           # Contextual ROI extraction module
│   └── trainer.py                 # Multi-epoch optimization & loss functions
├── demo.py                        # Standalone end-to-end verification script
├── main.py                        # Central command-line interface (CLI / Web / Train)
├── train.py                       # High-performance model training script
├── requirements.txt               # Complete Python package specifications
└── README.md                      # Comprehensive project documentation
```

---

## 5. How to Run the Project

### Prerequisites
Activate the project's virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Option 1: Run the Interactive Web CAD Workstation (Recommended)
Launch the web interface:
```bash
python main.py --web
```
Then open your web browser at:
`http://127.0.0.1:5000`

**Features in the Web App:**
* **Upload MRI Scans**: Upload custom MRI images or pick from preloaded Glioma, Meningioma, Pituitary, and Non-Tumor cases.
* **Instant Synthetic Scan Generator**: Replicate any MRI cross-section (Axial, Coronal, Sagittal) across Grade IV Glioblastoma, Meningioma, Pituitary, or Normal Brain.
* **Multi-Stage Visual Diagnostics**: Inspect Input MRI &rarr; CLAHE Enhancement &rarr; YOLO-NAS Bounding Box &rarr; Extracted ROI &rarr; Posterior Class Probabilities.
* **Benchmarks Tab**: Inspect Table 1 data and Figure 7 comparison graph.
* **Confusion Matrix Tab**: Inspect Figure 6 heatmap with quantitative sensitivity/specificity breakdowns.
* **Printable Medical Report**: 1-click printable clinical summary.

### Option 2: Run End-to-End CLI Pipeline Demonstration
To verify all stages from the terminal and generate Figure 6 and Figure 7:
```bash
python main.py --demo
# or
python demo.py
```

### Option 3: Retrain Models
To train both YOLO-NAS and EfficientNet on MRI datasets:
```bash
python main.py --train --epochs 12
```

---

## 6. References
1. Babu Vimala, Baiju, et al. "Detection and classification of brain tumor using hybrid deep learning models." Scientific reports 13.1 (2023): 23029.
2. Saeedi, Soheila, et al. "MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques." BMC Medical Informatics and Decision Making 23.1 (2023): 16.
3. Almufareh, Maram Fahaad, et al. "Automated brain tumor segmentation and classification in MRI using YOLO-based deep learning." IEEE Access 12 (2024): 16189-16207.
4. Mithun, M. S., and S. Joseph Jawhar. "Detection and classification on MRI images of brain tumor using YOLO NAS deep learning model." Journal of Radiation Research and Applied Sciences 17.4 (2024): 101113.
5. Rahman, Takowa, and Md Saiful Islam. "MRI brain tumor detection and classification using parallel deep convolutional neural networks." Measurement: Sensors 26 (2023): 100694.
