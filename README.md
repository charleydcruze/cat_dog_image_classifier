# Cat & Dog Image Classifier (Full-Stack Explainable AI & PWA)

A production-ready, full-stack computer vision application built with **PyTorch**, **FastAPI**, and a **Progressive Web App (PWA)** frontend. This application executes real-time binary image classification (Cat vs. Dog) powered by a deep convolutional neural network (ResNet-18), computes visual **Gradient-weighted Class Activation Mapping (Grad-CAM)** heatmaps for model interpretability, and generates downloadable **PDF Diagnostic Reports** via `fpdf2`.

---

## 1. Overview & Live System Architecture

The **Cat & Dog Image Classifier** bridges deep learning research with robust web engineering. Standard deep neural networks frequently operate as opaque "black boxes," obscuring whether predictions depend on true anatomical features or background artifacts. This project incorporates Explainable AI (XAI) directly into the user workflow alongside server-side PDF document generation.

The complete system architecture operates across three main layers:

1. **Frontend Layer (GitHub Pages PWA)**: A responsive single-page web interface providing drag-and-drop image uploading, webcam capture, dynamic result rendering, and offline asset caching (`serviceWorker.js`).
2. **Inference & Explainability Layer (FastAPI on Render)**: An asynchronous REST API hosted on Render that preprocesses image tensors, executes PyTorch forward passes, extracts feature map gradients, and generates color-mapped Grad-CAM visual overlays.
3. **Reporting Layer (fpdf2)**: An automated document generation service embedded in the backend that encodes metrics, class IDs, and decoded Base64 Grad-CAM heatmaps into downloadable A4 PDF reports.

### Live Cloud Deployments
* **PWA Frontend**: Hosted live on GitHub Pages.
* **Production API Backend**: Hosted live on Render.

---

## 2. Key Features

* **ResNet-18 Deep Learning Inference**: High-precision binary image classification leveraging PyTorch transfer learning and normalized image tensor pipelines.
* **Explainable AI (Grad-CAM)**: Interactive visual explanations that project activation gradients onto input images, highlighting the exact visual features (such as ears, eyes, and snout) driving classification decisions.
* **Automated PDF Report Generation**: One-click download of styled diagnostic reports containing classification labels, confidence percentages, ImageNet class IDs, and embedded high-resolution Grad-CAM heatmap overlays.
* **Memory-Optimized Cloud Deployment**: Engineered for low-resource environments using CPU-only PyTorch builds (`whl/cpu`), custom Matplotlib configuration (`MPLCONFIGDIR=/tmp`), and an asynchronous Uvicorn ASGI server.
* **Progressive Web App (PWA)**: Mobile-friendly frontend equipped with service worker caching (`serviceWorker.js`), web manifest metadata (`manifest.json`), and asynchronous `fetch` integration.

---

## 3. Tech Stack & Environment Configuration

### Software & Framework Breakdown

| Component | Technology | Role & Usage |
| :--- | :--- | :--- |
| **Machine Learning Core** | PyTorch, torchvision | ResNet-18 model loading, forward inference, tensor transforms (`transforms.Normalize`). |
| **Explainable AI** | `pytorch-grad-cam` | Gradient computation with respect to final convolutional layer target activations. |
| **Backend Web Server** | FastAPI, Uvicorn | Asynchronous HTTP REST endpoints (`/predict`, `/download-report`), CORS management, Pydantic validation. |
| **Document Engine** | `fpdf2`, Pillow (PIL) | PDF layout generation, Base64 image stream decoding, and memory buffer formatting. |
| **Computer Vision** | OpenCV (`opencv-python-headless`) | Color mapping (`COLORMAP_JET`), spatial resizing, and image tensor manipulation. |
| **Frontend PWA** | HTML5, CSS3, JavaScript (ES6+) | Drag-and-drop interface, camera API integration, async `fetch` handlers, Service Worker offline caching. |
| **Cloud Hosting** | Render (Backend), GitHub Pages (Frontend) | Serverless container deployment (Linux CPU) and static site delivery. |

### Environment Variables (Render Backend)
To ensure optimal performance and prevent out-of-memory (`512Mi`) crashes on cloud container instances, set the following environment variables:

```bash
MPLCONFIGDIR=/tmp       # Redirects Matplotlib font cache building to temporary disk storage
PYTHONOPTIMIZE=1         # Enables basic Python runtime code optimizations
PORT=10000               # Hardcodes default Uvicorn listening port for Render routing
```

---

## 4. End-to-End System Workflow & API Reference

### System Data Flow

1. **Upload & Preprocessing**: The user selects an image or captures a photo via webcam on the PWA frontend.
2. **Inference Request (`POST /predict`)**: The image file is submitted via `FormData` to the FastAPI backend.
3. **ResNet-18 & Grad-CAM Analysis**: PyTorch computes binary probabilities while Grad-CAM calculates activation gradients on layer4.
4. **Interactive Response**: FastAPI returns JSON containing classification label, confidence score, class ID, and Base64 heatmap data.
5. **PDF Report Export (`POST /download-report`)**: Upon clicking the download button, the frontend sends the prediction payload to `/download-report`, where `fpdf2` dynamically generates and streams a styled PDF document.

---

### REST API Specification

#### 1. Image Prediction Endpoint
* **URL**: `/predict`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Payload**: `file` (Image file: JPEG/PNG)
* **Response Status**: `200 OK`
* **Response JSON Structure**:
```json
{
  "label": "Cat",
  "confidence": 0.9852,
  "class_id": 281,
  "heatmap": "data:image/jpeg;base64,..."
}
```

#### 2. PDF Diagnostic Report Endpoint
* **URL**: `/download-report`
* **Method**: `POST`
* **Content-Type**: `application/json`
* **Payload JSON Structure**:
```json
{
  "label": "Cat",
  "confidence": 0.9852,
  "class_id": 281,
  "heatmap": "data:image/jpeg;base64,..."
}
```
* **Response Status**: `200 OK`
* **Response Header**: `Content-Disposition: attachment; filename="Cat_Classification_Report.pdf"`
* **Content-Type**: `application/pdf` (Binary Stream)

---

## 5. Repository Structure

| File / Directory | Component | Description |
| :--- | :--- | :--- |
| `backend.py` | FastAPI Backend | Asynchronous API defining `/predict` and `/download-report` routes, loading ResNet-18 weights, computing Grad-CAM, and generating PDF documents. |
| `cat_dog_resnet18.pth` | Model Checkpoint | Fine-tuned PyTorch ResNet-18 binary classification weights. |
| `train.py` | PyTorch Training | Script for dataset training, data augmentation, loss calculation, and weight optimization. |
| `sort_and_balance.py` | Utility Script | Organizes image samples into class directories and balances sample distributions. |
| `get_clean_data.py` | Data Cleaning | Scans image collections, checks JPEG/PNG headers, and discards unreadable or corrupt files. |
| `requirements.txt` | Package Manifest | CPU-optimized PyTorch dependencies, FastAPI, `fpdf2`, `opencv-python-headless`, and Uvicorn. |
| `index.html` | Frontend UI | HTML5 PWA single-page interface with image upload, live camera feed, results card, and PDF download button. |
| `serviceWorker.js` | PWA Caching | Progressive Web App worker handling offline asset caching and network strategy. |
| `manifest.json` | PWA Metadata | Web application configuration defining icons, splash colors, and display mode. |
| `CSS/` | Styling | Responsive layout stylesheets for mobile, tablet, and desktop viewports. |

---

## 6. Local Development & Installation Guide

### Prerequisites
* **Python**: Version 3.9, 3.10, or 3.11 recommended.
* **Git**: Installed on your system.

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd cat_dog_image_classifier
```

### Step 2: Create & Activate Virtual Environment
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

### Step 3: Install Lightweight Dependencies
Install the required packages using CPU-only PyTorch indices to save disk space and RAM:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Data Preparation & Model Training (Optional)
If training from scratch or updating model weights:
```bash
# 1. Verify raw dataset integrity
python3 get_clean_data.py

# 2. Balance class distributions
python3 sort_and_balance.py

# 3. Train ResNet-18 model
python3 train.py
```

### Step 5: Launch FastAPI Backend Server
Run the local backend server using Uvicorn:
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```
The local API will be accessible at `http://localhost:8000`.

### Step 6: Launch Frontend PWA
Serve the static web app files using Python's built-in HTTP server:
```bash
python3 -m http.server 8080
```
Open `http://localhost:8080` in your web browser to test predictions and PDF exports locally.

---

## 7. Deep-Dive: Explainable AI & PDF Reporting Mechanics

### Grad-CAM Interpretability Pipeline
1. **Target Feature Capture**: Extracts feature maps from the final convolutional layer of ResNet-18 (`layer4[1].conv2`).
2. **Gradient Backpropagation**: Computes the gradients of the predicted class score with respect to activation maps.
3. **Global Average Pooling**: Calculates weight values representing the importance of each feature map for the target class.
4. **ReLU Activation & Color Mapping**: Applies a ReLU activation function to retain positive features and projects the heatmap using OpenCV's `COLORMAP_JET`.

### Automated PDF Report Generation (`fpdf2`)
When `/download-report` is invoked:
1. **Pydantic Deserialization**: Parses JSON payload into a structured `ReportData` object containing class label, confidence percentage, ImageNet class ID, and Base64 heatmap string.
2. **Base64 Decoding**: Decodes the heatmap image string back into binary JPEG bytes using `base64.b64decode()` and loads it into an in-memory `io.BytesIO` stream.
3. **FPDF Canvas Assembly**: Formats document margins, titles, prediction metadata, and renders the heatmap image centered at 140mm width.
4. **Binary Streaming**: Converts the document output into Latin-1 encoded bytes and streams it via `fastapi.Response` with `application/pdf` headers.

---

## 8. Author & License

* **Developer**: charleydcruze
* **License**: Open-source under the MIT License.
