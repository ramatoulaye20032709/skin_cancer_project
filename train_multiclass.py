# ==============================================================
#   PROJET : Détection des tumeurs de la peau (ML/DL)
#   SCRIPT D'ENTRAÎNEMENT — Modèle Multiclasse HAM10000
#   7 classes de lésions cutanées
#   Lancement : python train_multiclass.py
# ==============================================================

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────
BASE_DIR    = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
DATA_DIR    = BASE_DIR / "data" / "ham10000"
IMAGES_DIR  = DATA_DIR / "images"
CSV_PATH    = DATA_DIR / "GroundTruth.csv"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
IMG_SIZE    = 64
BATCH_SIZE  = 64
EPOCHS      = 20
LEARNING_RATE = 0.001

# ── Classes HAM10000 ──────────────────────────────────────
CLASSES = {
    'MEL'  : 'Melanoma',
    'NV'   : 'Nevus',
    'BCC'  : 'Basal Cell Carcinoma',
    'BKL'  : 'Benign Keratosis',
    'DF'   : 'Dermatofibroma',
    'VASC' : 'Vascular Lesion',
    'AKIEC': 'Actinic Keratosis'
}
CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("   ENTRAÎNEMENT MODÈLE MULTICLASSE — HAM10000")
print("=" * 60)
print(f"✓ Dataset    : {DATA_DIR}")
print(f"✓ Images     : {IMAGES_DIR}")
print(f"✓ Classes    : {NUM_CLASSES}")
print(f"✓ IMG_SIZE   : {IMG_SIZE}x{IMG_SIZE}")
print(f"✓ BATCH_SIZE : {BATCH_SIZE}")
print(f"✓ EPOCHS     : {EPOCHS}")
print("=" * 60)

# ── Charger le CSV ────────────────────────────────────────
print("\n[1/6] Chargement du CSV...")
df = pd.read_csv(CSV_PATH)
print(f"✓ CSV chargé : {len(df)} lignes")
print(f"✓ Colonnes   : {list(df.columns)}")

# Trouver la colonne image et label
print(df.head())

# Identifier les colonnes
img_col = df.columns[0]  # première colonne = image_id

# Créer colonne label (la classe avec valeur 1)
df['label'] = df[CLASS_NAMES].idxmax(axis=1)
df['label_idx'] = df['label'].map({c: i for i, c in enumerate(CLASS_NAMES)})

# Ajouter le chemin complet de l'image
df['filepath'] = df[img_col].apply(
    lambda x: str(IMAGES_DIR / f"{x}.jpg")
)

# Vérifier que les images existent
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
print(f"✓ Images trouvées : {len(df)}")

# Distribution des classes
print("\n── Distribution des classes ──")
for cls, name in CLASSES.items():
    count = (df['label'] == cls).sum()
    print(f"  {cls:6s} | {name:30s} | {count:5d} images")

# ── Split train/val/test ──────────────────────────────────
print("\n[2/6] Split train/val/test...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
)

print(f"✓ Train : {len(train_df)} images")
print(f"✓ Val   : {len(val_df)} images")
print(f"✓ Test  : {len(test_df)} images")

# Sauvegarder les splits
train_df.to_csv(BASE_DIR / "data" / "train_multiclass.csv", index=False)
val_df.to_csv(BASE_DIR / "data" / "val_multiclass.csv", index=False)
test_df.to_csv(BASE_DIR / "data" / "test_multiclass.csv", index=False)

# ── Class weights ─────────────────────────────────────────
print("\n[3/6] Calcul des class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_df['label_idx'].values
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("✓ Class weights :")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:6s} : {class_weights[i]:.3f}")

# ── Data Pipeline ─────────────────────────────────────────
print("\n[4/6] Construction du pipeline de données...")

def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_dataset(dataframe, augment_data=False, shuffle=False):
    filepaths = dataframe['filepath'].values
    labels    = dataframe['label_idx'].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=42)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, augment_data=True, shuffle=True)
val_ds   = make_dataset(val_df,   augment_data=False, shuffle=False)
test_ds  = make_dataset(test_df,  augment_data=False, shuffle=False)

print("✓ Datasets créés")

# ── Construction du modèle ────────────────────────────────
print("\n[5/6] Construction du modèle CNN multiclasse...")

def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Bloc 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Bloc 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Bloc 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Bloc 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Classificateur
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="CNN_Multiclass")
    return model

model = build_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "meilleur_modele_multiclass.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        str(MODEL_DIR / "training_log.csv")
    )
]

# ── Entraînement ──────────────────────────────────────────
print("\n[6/6] Entraînement en cours...")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ── Évaluation ────────────────────────────────────────────
print("\n── Évaluation sur le test set ──")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"✓ Test Accuracy : {test_acc*100:.2f}%")
print(f"✓ Test Loss     : {test_loss:.4f}")

# Rapport de classification
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n── Rapport de classification ──")
print(classification_report(
    y_true, y_pred,
    target_names=[CLASSES[c] for c in CLASS_NAMES]
))

# ── Courbes d'entraînement ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],    label='Train')
axes[0].plot(history.history['val_accuracy'],label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],    label='Train')
axes[1].plot(history.history['val_loss'],label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Courbes d\'entraînement — CNN Multiclasse HAM10000', fontsize=14)
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "courbes_entrainement.png"), dpi=150)
plt.close()
print(f"\n✓ Courbes sauvegardées : {MODEL_DIR / 'courbes_entrainement.png'}")

# ── Matrice de confusion ──────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[CLASSES[c] for c in CLASS_NAMES],
    yticklabels=[CLASSES[c] for c in CLASS_NAMES]
)
plt.title('Matrice de Confusion')
plt.ylabel('Vrai label')
plt.xlabel('Prédit')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "matrice_confusion.png"), dpi=150)
plt.close()
print(f"✓ Matrice de confusion sauvegardée")

# ── Sauvegarder les infos du modèle ──────────────────────
import json
model_info = {
    "classes"     : CLASS_NAMES,
    "class_names" : CLASSES,
    "img_size"    : IMG_SIZE,
    "num_classes" : NUM_CLASSES,
    "test_accuracy": float(test_acc)
}
with open(str(MODEL_DIR / "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)
print(f"✓ Infos modèle sauvegardées : model_info.json")

print("\n" + "=" * 60)
print("   ENTRAÎNEMENT TERMINÉ !")
print(f"   Modèle sauvegardé : models/meilleur_modele_multiclass.keras")
print(f"   Test Accuracy     : {test_acc*100:.2f}%")
print("=" * 60)
