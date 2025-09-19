# 🥔 Potato Diseases Prediction using Deep Learning

> 🌿 A deep learning-powered web app to detect **7 potato conditions**:  
**Black Scurf, Blackleg, Common Scab, Dry Rot, Healthy, Miscellaneous, Pink Rot**  

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" />
  <img src="https://img.shields.io/badge/TensorFlow/Keras-✅-orange" />
  <img src="https://img.shields.io/badge/Streamlit-App-red" />
  <img src="https://img.shields.io/badge/Dataset-Potato%20Images-green" />
</p>

---

## 🔥 Project Overview

This project leverages **Convolutional Neural Networks (CNNs)** to classify potato diseases from leaf/tuber images.  
The app provides **confidence scores** and **disease classification results** in a user-friendly interface built with **Streamlit**.

---

## 📸 App Demo

### 1️⃣ Upload an Image  
The user uploads a potato image to the web app.  
<p align="center">
  <img src="Screenshot 2025-09-19 171200.png" width="500" />
</p>

---

### 2️⃣ Image Preview Before Detection  
Once uploaded, the app shows a **preview of the selected image** for confirmation.  
<p align="center">
  <img src="Screenshot 2025-09-19 171252.png" width="400" />
</p>

---

### 3️⃣ Disease Detection Results  
The model predicts the class with a **confidence score**.  
<p align="center">
  <img src="assets/ui_results.png" width="450" />
</p>

- ✅ **Label:** Healthy / Disease name  
- 📊 **Confidence Level:** e.g., 99.95%  
- 💡 **Recommendation:** Guidance for farmers (e.g., “Great news! Your potato is healthy”).  

---

## 🧩 Detectable Diseases

- 🟤 Black Scurf  
- ⚫ Blackleg  
- 🤎 Common Scab  
- 🟠 Dry Rot  
- 🟢 Healthy Potatoes  
- 🟡 Miscellaneous  
- 🌸 Pink Rot  

---

## ⚡ How It Works

1. Upload a potato leaf/tuber image.  
2. Preprocessing (resize to 256x256).  
3. Model predicts one of 7 classes.  
4. Confidence score + recommendation displayed in UI.  

---

## 📊 Example Output

**Input Image → Prediction:**  
| Input | Prediction | Confidence |  
|-------|-------------|-------------|  
| Potato Leaf | 🟢 Healthy | 99.95% |  
| Potato Tuber | 🟤 Black Scurf | 96.22% |  

---

## 🚀 Try It Out

Run locally:  
```bash
streamlit run app.py
