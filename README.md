[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict)
# 👁️ Retina-CVD Predict
### **Dual-Stream Hybrid Ensemble for Multi-Organ Risk Assessment**

**Retina-CVD Predict** is an Explainable AI (XAI) platform designed to predict the 5-year risk of **Stroke**, **Chronic Kidney Disease (CKD)**, and **Hypertension** through non-invasive retinal fundus imaging. By utilizing the retina as a "window" into systemic microvascular health, this system identifies pathological biomarkers often invisible to human clinicians.

---

## 🚀 Core Technology
The system operates on a **Dual-Stream Hybrid Ensemble** architecture:
* **The Generalist (ResNet-50):** Optimized for global feature extraction to identify macro-structural changes associated with **Stroke** and **CKD**.
* **The Specialist (EfficientNet-B3):** A high-resolution stream dedicated to **Hypertension**, focusing on fine-grained arteriolar narrowing and vascular density.
* **Cubic XAI ($x^3$):** Integrated Grad-CAM visualization using **Cubic Saliency Sharpening** to eliminate background noise and isolate critical vascular lesions.

## ✨ Key Features
* **Dynamic Clinical Triage:** Real-time risk stratification using medical-grade color coding (Green/Yellow/Red).
* **Probability-Scaled Heatmaps:** Saliency intensity is mathematically tied to the prediction score, ensuring visual honesty in the diagnostic output.
* **Automated Interpretation:** A heuristic engine that generates professional insights, such as detecting "vessel tortuosity" or "micro-aneurysm patterns."
* **Modern UI/UX:** A responsive dashboard optimized for clinical environments, featuring synchronized triple-pane evidence views.

## 📊 Dataset & Training
The models were developed using a curated subset of high-resolution retinal fundus scans:
* **Primary Source:** **ODIR-5K**(Dataset of 5000+ fundal images with labels) for clinical validation.
* **Preprocessing:** Images were normalized using **Ben Graham's Preprocessing** and resized to $224 \times 224$ (ResNet) and $300 \times 300$ (EfficientNet).
* **Storage:** Large model weights ($>100\text{MB}$) are managed via **Git LFS** for repository integrity.

## 🛠️ Installation & Setup
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Abir71073/Retina-CVD-Predict.git](https://github.com/Abir71073/Retina-CVD-Predict.git)
    cd Retina-CVD-Predict
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Dashboard:**
    ```bash
    python app.py
    ```
    *Access the UI at `http://127.0.0.1:5000`.*

## 👥 Contributors
* **Abir Kumar Chakraborty**
* **Institution:** Institute of Engineering and Management (IEM), Kolkata
* **Guided by:** Prof. Shreejita Mukherjee

---
*Disclaimer: Developed for the Innovative Project-II curriculum. For research and academic purposes only. Not intended for direct diagnostic use.*
