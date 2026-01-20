# 🔧 PCB Defect Detection and Classification System

## 📌 Overview

This project presents an **automated PCB Defect Detection and Classification System** that combines **image processing** and **deep learning** to identify, localize, and classify defects in Printed Circuit Boards (PCBs).

The system compares a defect-free template PCB with a test PCB, extracts defect regions, and classifies them using a trained **EfficientNet** model. A **Streamlit web application** allows users to upload images, visualize results in real time, and download outputs.

---

## 🎯 Objectives

The system aims to:

* Detect and localize PCB defects using image subtraction and contour analysis
* Classify defects into predefined categories using a CNN model
* Provide a user-friendly web interface for real-time inference
* Generate annotated images and downloadable logs
* Deliver a complete, deployable end-to-end solution

---

## 📂 Dataset

* **Dataset Used:** DeepPCB Dataset
* Contains paired images of:

  * **Template PCB (defect-free)**
  * **Test PCB (with defects)**

---

## 🧱 Project Structure
```
PCB_Project/
│
├── Milestone1/
│   ├── data/
│   ├── outputs/
│   └── scripts/
│
├── Milestone2/
│   ├── data/
│   ├── models/
│   ├── results/
│   └── training scripts
│
├── Milestone3/
│   ├── app.py
│   ├── inference.py
│   ├── utils.py
│   ├── models/
│   └── requirements.txt
│
└── README.md
```

---

## ✅ Milestone Summary

### **Milestone 1 — Image Processing**

* Image alignment and preprocessing
* Template-based image subtraction
* Thresholding and morphological filtering
* Defect mask generation
* Contour detection and ROI extraction

### **Milestone 2 — Model Training & Evaluation**

* Trained **EfficientNet-B0** model in PyTorch
* Image size: **128×128**
* Optimizer: **Adam**
* Loss: **Cross-Entropy / Focal Loss**
* Achieved **~97.9% test accuracy**
* Generated:

  * Accuracy plot
  * Loss plot
  * Confusion matrix
  * Classification report

### **Milestone 3 — Web UI + Backend Integration**

* Streamlit-based web application
* Upload template and test images
* Real-time defect detection
* Annotated PCB output with bounding boxes and labels
* Modular backend with:

  * `utils.py` → image processing
  * `inference.py` → model prediction

### **Milestone 4 — Finalization & Deployment**

* Downloadable:

  * Annotated image
  * Prediction log (CSV)
* Fully functional Streamlit app
* GitHub repository organized
* Ready for presentation and demo

---

## 🧠 Defect Classes

The model classifies defects into six categories:

* Missing Hole
* Mouse Bite
* Open Circuit
* Short
* Spur
* Spurious Copper

---

## 🚀 How to Run Locally

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/balaganeshbalireddi/Pcb_Project.git
cd Pcb_Project/Milestone3
```

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit app**

```bash
streamlit run app.py
```

Then open in your browser:

```
http://localhost:8501
```

---

## 🌐 Deployment

The app is designed to be deployed on **Streamlit Cloud**.
Use `Milestone3/app.py` as the entry point and ensure `requirements.txt` includes:

```
streamlit
torch
torchvision
opencv-python-headless
numpy
pandas
pillow
```

---

## 📈 Performance

* Test Accuracy: **97.91%**
* Fast inference: typically **< 3 seconds per image pair**
* Low false positives and false negatives

---

## 📌 Future Enhancements

* Add defect severity scoring
* Support batch uploads
* Real-time camera integration
* Edge deployment on embedded systems

