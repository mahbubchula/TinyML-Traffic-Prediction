# 🚦 Adaptive TinyML for Edge-Based Traffic Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinyml-traffic-prediction.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange.svg)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Few-Shot Learning framework for Intelligent Transportation Systems (ITS) that adapts to new traffic environments using <6KB of memory.**

---

## 👥 Authors
- **Mahbub Hassan** (Chulalongkorn University)
- **Md Maruf Hassan** (Southeast University)
- **Touhid Bhuiyan** (Washington University of Science and Technology)
*Department of Civil Engineering, Chulalongkorn University*

---

## 📄 Abstract
The deployment of Intelligent Transportation Systems (ITS) is often hindered by high computational costs and data scarcity. This project introduces a **Few-Shot TinyML framework** capable of accurate traffic forecasting on resource-constrained edge devices. 

Our **1D-CNN model** achieves a memory footprint of just **5.46 KB** and adapts to new traffic environments using only **48 hours** of calibration data, effectively solving the "Cold Start" problem for newly installed traffic sensors.

### 🖼️ System Architecture
![Methodology](https://github.com/mahbubchula/TinyML-Traffic-Prediction/blob/main/paper/Fig1_Methodology.png?raw=true)
*(Figure 1: Proposed Edge-Adaptive Framework connecting Source Domain, Quantization, and Target Domain Adaptation)*

---

## 🏆 Key Research Results

| Metric | Result | Impact |
| :--- | :--- | :--- |
| **📉 Model Size** | **5.46 KB** | **90% reduction** vs. standard CNNs. Fits on Arduino Nano 33 BLE. |
| **🎯 Accuracy (MSE)** | **564.72** | High accuracy achieved after just **48 hours** of Few-Shot adaptation. |
| **⚡ Energy Efficiency** | **2.5 mJ** | **60x more efficient** per inference compared to Cloud/Wi-Fi transmission. |

### 📊 Performance Visualization
![Results](https://github.com/mahbubchula/TinyML-Traffic-Prediction/blob/main/paper/Fig3_Results_Comparison.png?raw=true)
*(Figure 3: Comparison of Baseline (Blue) vs. Adapted TinyML Model (Green) against Ground Truth)*

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/mahbubchula/TinyML-Traffic-Prediction.git](https://github.com/mahbubchula/TinyML-Traffic-Prediction.git)
cd TinyML-Traffic-Prediction
