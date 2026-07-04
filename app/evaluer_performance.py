"""
==============================================================
DermIA — Script d'évaluation des performances du modèle
==============================================================
Calcule accuracy, précision, rappel, F1-score, AUC-ROC et la
matrice de confusion sur le VRAI jeu de TEST (10% d'images
jamais vues à l'entraînement), puis sauvegarde les résultats
pour qu'ils s'affichent dans l'onglet "📊 Performance du modèle"
de app.py.

MÉTHODE : ce script ne lit PAS un dossier "test" séparé. Il
reconstruit le jeu de test en reproduisant EXACTEMENT le même
split que train_v5_lr_corrige.py (même random_state=42, mêmes
proportions 80/10/10, même stratification). Comme le random_state
est fixé, le découpage obtenu est identique à celui utilisé
pendant l'entraînement : les images sélectionnées ici n'ont
jamais servi à entraîner ni à valider le modèle, ce qui garantit
des métriques fiables (pas de fuite de données).

Lancement (depuis Anaconda Prompt, dans le dossier du projet) :
    python evaluer_performance.py
==============================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif, juste sauvegarde de fichiers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# ── Chemins (identiques à train_v5_lr_corrige.py) ────────────
BASE_DIR   = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
MODEL_PATH = BASE_DIR / "models" / "meilleur_modele_v3.keras"
DATA_DIR   = BASE_DIR / "data" / "ham10000"
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH   = DATA_DIR / "GroundTruth.csv"
OUT_DIR    = BASE_DIR / "data"
IMG_SIZE   = 224

# Même ordre que CLASS_NAMES dans train_v5_lr_corrige.py et que CLASSES dans app.py
CODES = ['MEL', 'NV', 'BCC', 'BKL', 'DF', 'VASC', 'AKIEC']


def construire_jeu_test():
    """Reproduit exactement le split de train_v5_lr_corrige.py pour récupérer
    le même 10% d'images 'test' (jamais vues, ni en train, ni en validation)."""
    df = pd.read_csv(CSV_PATH)
    img_col = df.columns[0]
    df['label'] = df[CODES].idxmax(axis=1)
    df['label_idx'] = df['label'].map({c: i for i, c in enumerate(CODES)})
    df['filepath'] = df[img_col].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
    df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

    # Mêmes paramètres que dans train_v5_lr_corrige.py (lignes 86-87)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"   Split reproduit -> Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")
    return test_df


def charger_jeu_test():
    test_df = construire_jeu_test()
    images, labels = [], []
    for _, row in test_df.iterrows():
        try:
            img = Image.open(row['filepath']).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32))
            labels.append(int(row['label_idx']))
        except Exception as e:
            print(f"   Image ignorée ({row['filepath']}) : {e}")

    print("\n   Répartition du jeu de test par classe :")
    labels_arr = np.array(labels)
    for i, code in enumerate(CODES):
        print(f"   {code} : {int(np.sum(labels_arr == i))} images")

    return np.array(images), labels_arr


def main():
    print("== DermIA — Évaluation des performances ==\n")

    print("1) Chargement du modèle...")
    if not MODEL_PATH.exists():
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("\n2) Reconstruction du jeu de test (10% hold-out, jamais vu à l'entraînement)...")
    if not CSV_PATH.exists():
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        return

    X_test, y_test = charger_jeu_test()
    print(f"\n   Total : {len(X_test)} images de test.")

    if len(X_test) == 0:
        print("❌ Aucune image de test trouvée. Vérifie DATA_DIR/IMAGES_DIR/CSV_PATH.")
        return

    print("\n3) Prédiction sur le jeu de test...")
    y_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_proba, axis=1)

    print("\n4) Calcul des métriques (classification_report)...")
    rapport = classification_report(
        y_test, y_pred, target_names=CODES, output_dict=True, zero_division=0
    )
    accuracy_globale = float(rapport["accuracy"])
    print(f"   Accuracy globale : {accuracy_globale * 100:.2f}%")

    print("\n5) Calcul de l'AUC-ROC (one-vs-rest) et tracé des courbes ROC...")
    classes_presentes = sorted(set(y_test.tolist()))
    y_test_bin = label_binarize(y_test, classes=list(range(len(CODES))))
    auc_par_classe = {}

    plt.figure(figsize=(7, 6))
    for i, code in enumerate(CODES):
        if i not in classes_presentes or len(np.unique(y_test_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        auc_par_classe[code] = float(auc_i)
        plt.plot(fpr, tpr, label=f"{code} (AUC = {auc_i:.2f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Hasard (AUC = 0.50)")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC par classe — DermIA")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close()

    auc_macro = float(np.mean(list(auc_par_classe.values()))) if auc_par_classe else None
    if auc_macro:
        print(f"   AUC-ROC macro : {auc_macro * 100:.2f}%")

    print("\n6) Calcul et tracé de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CODES))))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Greens")
    plt.title("Matrice de confusion — DermIA")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.xticks(range(len(CODES)), CODES, rotation=45)
    plt.yticks(range(len(CODES)), CODES)
    seuil = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(CODES)):
        for j in range(len(CODES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > seuil else "black", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\n7) Sauvegarde des résultats (metriques_modele.json)...")
    resultats = {
        "date_evaluation": datetime_now_iso(),
        "nombre_images_test": int(len(X_test)),
        "methode": "Hold-out 10% reproduit depuis train_v5_lr_corrige.py (random_state=42, stratifié)",
        "accuracy_globale": accuracy_globale,
        "rapport_par_classe": rapport,
        "auc_roc_macro": auc_macro,
        "auc_roc_par_classe": auc_par_classe,
    }
    with open(OUT_DIR / "metriques_modele.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Terminé. Fichiers générés dans {OUT_DIR} :")
    print("   - metriques_modele.json")
    print("   - confusion_matrix.png")
    print("   - roc_curves.png")
    print("\nRelance ensuite `streamlit run app.py` et ouvre l'onglet "
          "'📊 Performance du modèle'.")


def datetime_now_iso():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()