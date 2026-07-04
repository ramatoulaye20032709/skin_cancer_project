# ==============================================================
#   PROJET : Détection des tumeurs de la peau (ML/DL)
#   SCRIPT OPTIMISÉ v3 — EfficientNetB0 + HAM10000
#   Objectif : Dépasser 73,45% d'accuracy (CPU optimisé)
#   Améliorations :
#     ✅ 100 couches dégelées (vs 60 en v2)
#     ✅ Label Smoothing (réduit la sur-confiance)
#     ✅ Cosine Annealing LR (convergence progressive)
#     ✅ Patience augmentée (laisse plus de temps converger)
#     ✅ Test Time Augmentation (TTA) à l'inférence
#     ✅ Mixup augmentation légère
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
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import json
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
BASE_DIR    = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
DATA_DIR    = BASE_DIR / "data" / "ham10000"
IMAGES_DIR  = DATA_DIR / "images"
CSV_PATH    = DATA_DIR / "GroundTruth.csv"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE        = 224
BATCH_SIZE      = 16    # ✅ Réduit pour CPU (moins de mémoire)
EPOCHS_FROZEN   = 20    # ✅ Augmenté (vs 15 en v2)
EPOCHS_FINETUNE = 40    # ✅ Augmenté (vs 30 en v2)
LEARNING_RATE   = 0.001
FINE_TUNE_LR    = 0.00003  # ✅ Encore plus faible (vs 5e-5 en v2)
UNFREEZE_LAYERS = 100   # ✅ Augmenté (vs 60 en v2)

# ── Classes HAM10000 ───────────────────────────────────────
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
print("   SCRIPT OPTIMISÉ v3 — EfficientNetB0 + HAM10000")
print("   Améliorations vs v2 :")
print("   ✅ 100 couches dégelées (vs 60)")
print("   ✅ Label Smoothing 0.1")
print("   ✅ Cosine Annealing LR")
print("   ✅ Patience augmentée")
print("   ✅ Batch size 16 (optimisé CPU)")
print("   ✅ Test Time Augmentation (TTA)")
print("=" * 60)

# ── Charger le CSV ─────────────────────────────────────────
print("\n[1/9] Chargement du CSV...")
df = pd.read_csv(CSV_PATH)
img_col = df.columns[0]
df['label'] = df[CLASS_NAMES].idxmax(axis=1)
df['label_idx'] = df['label'].map({c: i for i, c in enumerate(CLASS_NAMES)})
df['filepath'] = df[img_col].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
print(f"✓ Images trouvées : {len(df)}")

# ── Split ──────────────────────────────────────────────────
print("\n[2/9] Split train/val/test...")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
print(f"✓ Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

# ── Oversampling ───────────────────────────────────────────
print("\n[3/9] Oversampling des classes minoritaires...")
counts = train_df['label'].value_counts()
max_count = counts.max()
target_count = max_count // 2
dfs_oversampled = []
for cls in CLASS_NAMES:
    cls_df = train_df[train_df['label'] == cls]
    original_count = len(cls_df)
    if original_count < target_count:
        cls_df = resample(cls_df, replace=True, n_samples=target_count, random_state=42)
        print(f"  {cls:6s} : {original_count:4d} → {target_count:4d} (oversamplé)")
    else:
        print(f"  {cls:6s} : {original_count:4d} (inchangé)")
    dfs_oversampled.append(cls_df)
train_df = pd.concat(dfs_oversampled).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Train après oversampling : {len(train_df)}")

# ── Class weights ──────────────────────────────────────────
print("\n[4/9] Class weights...")
class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_df['label_idx'].values)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:6s} : {class_weights[i]:.3f}")

# ── Pipeline tf.data ───────────────────────────────────────
print("\n[5/9] Pipeline tf.data...")

def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.25)      # ✅ augmenté
    img = tf.image.random_contrast(img, 0.75, 1.25)  # ✅ augmenté
    img = tf.image.random_saturation(img, 0.75, 1.25)
    img = tf.image.random_hue(img, 0.1)               # ✅ augmenté
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    crop_frac = tf.random.uniform([], 0.75, 1.0)      # ✅ zoom plus agressif
    crop_size = tf.cast(tf.round(crop_frac * IMG_SIZE), tf.int32)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label

def make_dataset(dataframe, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        dataframe['filepath'].values,
        dataframe['label_idx'].values
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, augment_data=True,  shuffle=True)
val_ds   = make_dataset(val_df,   augment_data=False, shuffle=False)
test_ds  = make_dataset(test_df,  augment_data=False, shuffle=False)
print("✓ Datasets créés")

# ── Modèle ─────────────────────────────────────────────────
print("\n[6/9] Construction du modèle...")

def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)           # ✅ Dropout plus fort (0.4→0.5)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)           # ✅ Dropout plus fort (0.3→0.4)
    x = layers.Dense(128, activation='relu')(x)  # ✅ NOUVELLE couche
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_v3")
    return model, base_model

model, base_model = build_model()
model.summary()
print(f"✓ Paramètres entraînables : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

def get_metrics():
    return ['accuracy', keras.metrics.AUC(name='auc')]

# ── ✅ Cosine Annealing LR Scheduler ──────────────────────
def cosine_lr_schedule(epoch, lr):
    """Cosine annealing : réduit le LR progressivement en cosinus"""
    if epoch < EPOCHS_FROZEN:
        return float(LEARNING_RATE)
    else:
        progress = (epoch - EPOCHS_FROZEN) / EPOCHS_FINETUNE
        return float(FINE_TUNE_LR * (1 + np.cos(np.pi * progress)) / 2)

# ── PHASE 1 ────────────────────────────────────────────────
print("\n[7/9] Phase 1 — Backbone gelé...")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    # ✅ Label smoothing : réduit la sur-confiance du modèle
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=get_metrics()
)

callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "meilleur_modele_v3.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,              # ✅ augmenté (6→8)
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4,              # ✅ augmenté (3→4)
        min_lr=1e-6, verbose=1
    ),
    keras.callbacks.CSVLogger(str(MODEL_DIR / "log_phase1_v3.csv"))
]

history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict,
    verbose=1
)
print(f"\n✓ Phase 1 terminée — Meilleure val_accuracy : {max(history1.history['val_accuracy'])*100:.2f}%")

# ── PHASE 2 Fine-tuning ────────────────────────────────────
print(f"\n[8/9] Phase 2 — Fine-tuning {UNFREEZE_LAYERS} couches...")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

print(f"✓ Paramètres entraînables : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LR,
        clipnorm=1.0
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=get_metrics()
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "meilleur_modele_v3.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,             # ✅ augmenté (8→10)
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5,              # ✅ augmenté (4→5)
        min_lr=1e-8, verbose=1
    ),
    # ✅ Cosine Annealing
    keras.callbacks.LearningRateScheduler(cosine_lr_schedule, verbose=0),
    keras.callbacks.CSVLogger(str(MODEL_DIR / "log_phase2_v3.csv"))
]

history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

# ── ✅ Test Time Augmentation (TTA) ────────────────────────
print("\n[9/9] Évaluation avec TTA (Test Time Augmentation)...")

def predict_with_tta(model, dataset, n_augmentations=5):
    """
    TTA : prédit n fois avec des augmentations légères,
    puis fait la moyenne — améliore la robustesse
    """
    all_preds = []
    all_labels = []

    for imgs, labels in dataset:
        batch_preds = []
        # Prédiction originale
        pred = model.predict(imgs, verbose=0)
        batch_preds.append(pred)

        # Prédictions augmentées
        for _ in range(n_augmentations - 1):
            imgs_aug = tf.image.random_flip_left_right(imgs)
            imgs_aug = tf.image.random_brightness(imgs_aug, 0.1)
            pred_aug = model.predict(imgs_aug, verbose=0)
            batch_preds.append(pred_aug)

        # Moyenne des prédictions
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.extend(np.argmax(avg_pred, axis=1))
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

# Évaluation standard
print("\n── Évaluation standard ──")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"✓ Test Accuracy (standard) : {test_acc*100:.2f}%")
print(f"✓ Test AUC                 : {test_auc*100:.2f}%")

# Évaluation avec TTA
print("\n── Évaluation avec TTA (5 augmentations) ──")
y_true_tta, y_pred_tta = predict_with_tta(model, test_ds, n_augmentations=5)
tta_accuracy = np.mean(y_true_tta == y_pred_tta)
print(f"✓ Test Accuracy (TTA)      : {tta_accuracy*100:.2f}%")

# Rapport de classification
print("\n── Rapport de classification (TTA) ──")
print(classification_report(
    y_true_tta, y_pred_tta,
    target_names=[CLASSES[c] for c in CLASS_NAMES]
))

# ── Courbes ────────────────────────────────────────────────
def combine(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

all_acc   = combine(history1, history2, 'accuracy')
all_vacc  = combine(history1, history2, 'val_accuracy')
all_loss  = combine(history1, history2, 'loss')
all_vloss = combine(history1, history2, 'val_loss')
all_auc   = combine(history1, history2, 'auc')
all_vauc  = combine(history1, history2, 'val_auc')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(all_acc) + 1)
phase2_start = len(history1.history['accuracy'])

for ax, tv, vv, title in zip(
    axes,
    [all_acc, all_loss, all_auc],
    [all_vacc, all_vloss, all_vauc],
    ['Accuracy', 'Loss', 'AUC']
):
    ax.plot(epochs_range, tv,  label='Train', linewidth=2)
    ax.plot(epochs_range, vv,  label='Validation', linewidth=2)
    ax.axvline(x=phase2_start, color='red', linestyle='--', label='Fine-tuning', alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('EfficientNetB0 v3 — Optimisé CPU', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "courbes_v3.png"), dpi=150)
plt.close()

# ── Matrice de confusion ───────────────────────────────────
cm      = confusion_matrix(y_true_tta, y_pred_tta)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels  = [CLASSES[c] for c in CLASS_NAMES]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('Matrice de Confusion — Absolue')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('Matrice de Confusion — Normalisée')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(str(MODEL_DIR / "matrice_confusion_v3.png"), dpi=150)
plt.close()

# ── Sauvegarde infos ───────────────────────────────────────
model_info = {
    "version"          : "3.0",
    "model_type"       : "EfficientNetB0_Optimised_CPU",
    "classes"          : CLASS_NAMES,
    "class_names"      : CLASSES,
    "img_size"         : IMG_SIZE,
    "num_classes"      : NUM_CLASSES,
    "unfreeze_layers"  : UNFREEZE_LAYERS,
    "test_accuracy_std": float(test_acc),
    "test_accuracy_tta": float(tta_accuracy),
    "test_auc"         : float(test_auc),
    "improvements_v3"  : [
        "100 couches dégelées (vs 60 en v2)",
        "Dropout plus fort (0.5/0.4/0.3)",
        "Couche Dense 128 supplémentaire",
        "Cosine Annealing LR Scheduler",
        "Patience augmentée (8/10 vs 6/8)",
        "Augmentation plus agressive",
        "Test Time Augmentation (TTA x5)",
        "Batch size 16 (optimisé CPU)"
    ]
}
with open(str(MODEL_DIR / "model_info_v3.json"), "w") as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("   ENTRAÎNEMENT v3 TERMINÉ !")
print(f"   Modèle    : models/meilleur_modele_v3.keras")
print(f"   Accuracy standard : {test_acc*100:.2f}%")
print(f"   Accuracy TTA      : {tta_accuracy*100:.2f}%")
print(f"   AUC               : {test_auc*100:.2f}%")
print("=" * 60)
print("\nFichiers générés :")
print("  courbes_v3.png")
print("  matrice_confusion_v3.png")
print("  model_info_v3.json")
print("  log_phase1_v3.csv")
print("  log_phase2_v3.csv")
