# 🩺 DermIA — Détection Intelligente de Tumeurs Cutanées

> Projet de fin d'études — Licence Informatique | ESITEC Dakar, Sénégal

## 📌 Description

DermIA est un système intelligent de détection et classification de tumeurs cutanées basé sur le **Deep Learning** et le **Transfer Learning**. Il permet d'analyser des images dermoscopiques et de les classifier parmi 7 types de lésions cutanées.

## 🎯 Performances du modèle

| Métrique | Valeur |
|----------|--------|
| Accuracy | **80.64%** |
| AUC-ROC (macro) | **94.74%** |
| Dataset | HAM10000 (10 015 images) |
| Architecture | EfficientNetB0 + Transfer Learning |

## 🏷️ Classes détectées

| Code | Description |
|------|-------------|
| MEL | Mélanome |
| NV | Nævus mélanocytaire |
| BCC | Carcinome basocellulaire |
| BKL | Kératose bénigne |
| DF | Dermatofibrome |
| VASC | Lésion vasculaire |
| AKIEC | Kératose actinique |

## 🗂️ Structure du projet

## ✨ Fonctionnalités de l'application

- 📸 **Upload d'image** — Analyse d'une image dermoscopique
- 📦 **Batch upload** — Analyse de plusieurs images à la fois
- 🔥 **Grad-CAM** — Visualisation des zones d'attention du modèle
- 📊 **Métriques** — Affichage des performances du modèle
- 📄 **Export PDF** — Génération de rapport patient
- ✅ **Validation** — Filtre binaire pour vérifier si l'image est dermoscopique

## 🚀 Installation et lancement

```bash
# Cloner le dépôt
git clone https://github.com/ramatoulaye20032709/skin_cancer_project.git
cd skin_cancer_project

# Créer l'environnement Anaconda
conda create -n skin_cancer python=3.10
conda activate skin_cancer

# Installer les dépendances
pip install tensorflow streamlit opencv-python matplotlib scikit-learn reportlab

# Lancer l'application
streamlit run app/app.py
```

## 🛠️ Technologies utilisées

- **Python 3.10**
- **TensorFlow / Keras** — Deep Learning
- **EfficientNetB0** — Architecture pré-entraînée sur ImageNet
- **Streamlit** — Interface web
- **Grad-CAM** — Explicabilité du modèle
- **OpenCV** — Traitement d'images
- **Scikit-learn** — Métriques d'évaluation

## 👩‍💻 Auteure

**Ramatoulaye SY**
Étudiante en Licence Informatique — ESITEC Dakar, Sénégal

---
*Projet réalisé dans le cadre du mémoire de fin d'études — 2026*
