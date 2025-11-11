# ❤️ Heart Disease Prediction using XGBoost

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A **data science project** that predicts the likelihood of **heart disease** using the **UCI Heart Disease Dataset**.  
Built with **Python, XGBoost, and Streamlit**, it demonstrates an end-to-end ML pipeline — from preprocessing to deployment — focused on clinical interpretability.

---


## 🧠 Project Overview

This project predicts whether a patient has heart disease based on clinical attributes like:
- Age, Sex, Resting Blood Pressure  
- Serum Cholesterol & Maximum Heart Rate  
- Chest Pain Type & Exercise-Induced Angina  
- Thalassemia, ST Depression, and ECG Results  

### 🎯 Objectives
1. Build and tune a robust **classification model (XGBoost)**  
2. Prioritize **Recall** (catching every possible heart disease case)  
3. Interpret feature contributions using **feature importance**  
4. Deploy a **Streamlit web app** for live predictions  

---

## ⚙️ Tech Stack

| Layer | Tools |
|:------|:------|
| **Language** | Python |
| **ML / Modeling** | scikit-learn, XGBoost |
| **Visualization** | Matplotlib, SHAP |
| **App / Deployment** | Streamlit |
| **Data Source** | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) |

---

## 📈 Model Performance

| Metric | Score |
|:--------|:------:|
| **Accuracy** | 0.87 |
| **Recall (Heart Disease Detection)** | **0.82** |
| **ROC-AUC** | 0.91 |

🔍 The model is **optimized for recall**, ensuring it identifies most positive heart-disease cases — a critical factor in medical decision support.

---


---

## ⚡ Quickstart

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
bash'''
### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit App
streamlit run app.py


## ⚠️ Disclaimer

This project is for educational and research purposes only.
It is not a certified medical diagnostic tool and should not be used for clinical decision-making.


## 👨‍💻 Author

Ayush Dodal
Data Engineer | Data Scientist | Data Analyst | AI Engineer
ayushdodal1999@gmail.com
https://www.linkedin.com/in/ayush-dodal-b5646016b/
