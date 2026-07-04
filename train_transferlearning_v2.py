# ==============================================================
#   PROJET : Détection des tumeurs de la peau (ML/DL)
#   SCRIPT TRANSFER LEARNING v2 — EfficientNetB0 + HAM10000
#   Améliorations : augmentation, oversampling, plus de couches
#                   dégelées, métriques cliniques (AUC, Recall)
#   Lancement : python train_transferlearning_v2.py
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
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 30
LEARNING_RATE   = 0.001
FINE_TUNE_LR    = 0.00005
UNFREEZE_LAYERS = 60

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
print("   TRANSFER LEARNING v2 — EfficientNetB0 + HAM10000")
print("   Améliorations activées :")
print("   ✅ Augmentation + rotation/zoom (tf pures uniquement)")
print("   ✅ Oversampling classes minoritaires")
print(f"   ✅ {UNFREEZE_LAYERS} couches dégelées (vs 30 en v1)")
print("   ✅ Métriques AUC + Recall (priorité clinique)")
print("   ✅ Fine-tuning LR réduit à 5e-5")
print("=" * 60)
print(f"✓ Classes    : {NUM_CLASSES}")
print(f"✓ IMG_SIZE   : {IMG_SIZE}x{IMG_SIZE}")
print(f"✓ BATCH_SIZE : {BATCH_SIZE}")
print(f"✓ Phase 1    : {EPOCHS_FROZEN} epochs (gelé)")
print(f"✓ Phase 2    : {EPOCHS_FINETUNE} epochs (fine-tuning)")
print("=" * 60)

# ── Charger le CSV ─────────────────────────────────────────
print("\n[1/8] Chargement du CSV...")
df = pd.read_csv(CSV_PATH)
print(f"✓ CSV chargé : {len(df)} lignes")

img_col = df.columns[0]
df['label'] = df[CLASS_NAMES].idxmax(axis=1)
df['label_idx'] = df['label'].map({c: i for i, c in enumerate(CLASS_NAMES)})
df['filepath'] = df[img_col].apply(
    lambda x: str(IMAGES_DIR / f"{x}.jpg")
)
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
print(f"✓ Images trouvées : {len(df)}")

print("\n── Distribution des classes ──")
for cls, name in CLASSES.items():
    count = (df['label'] == cls).sum()
    print(f"  {cls:6s} | {name:30s} | {count:5d} images")

# ── Split ──────────────────────────────────────────────────
print("\n[2/8] Split train/val/test...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
)
print(f"✓ Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

train_df.to_csv(BASE_DIR / "data" / "train_multiclass.csv", index=False)
val_df.to_csv(BASE_DIR / "data"   / "val_multiclass.csv",   index=False)
test_df.to_csv(BASE_DIR / "data"  / "test_multiclass.csv",  index=False)

# ── Oversampling ───────────────────────────────────────────
print("\n[3/8] Oversampling des classes minoritaires...")
counts = train_df['label'].value_counts()
max_count = counts.max()
target_count = max_count // 2

dfs_oversampled = []
for cls in CLASS_NAMES:
    cls_df = train_df[train_df['label'] == cls]
    original_count = len(cls_df)
    if original_count < target_count:
        cls_df = resample(cls_df, replace=True, n_samples=target_count, random_state=42)
        print(f"  {cls:6s} : {original_count:4d} → {target_count:4d} images (oversamplé)")
    else:
        print(f"  {cls:6s} : {original_count:4d} images (inchangé)")
    dfs_oversampled.append(cls_df)

train_df = pd.concat(dfs_oversampled).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Train après oversampling : {len(train_df)} images")

# ── Class weights ──────────────────────────────────────────
print("\n[4/8] Calcul des class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_df['label_idx'].values
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:6s} : {class_weights[i]:.3f}")

# ── Data Pipeline ──────────────────────────────────────────
print("\n[5/8] Construction du pipeline...")

def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label

# ✅ CORRIGÉ — Uniquement des ops tf natives, zéro layer Keras
# Aucun if Python, aucun RandomRotation/Zoom/Translation
def augment(img, label):
    # Flips
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Couleur
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.08)

    # Zoom simulé : crop aléatoire + resize
    crop_frac = tf.random.uniform([], 0.80, 1.0)
    crop_size = tf.cast(tf.round(crop_frac * IMG_SIZE), tf.int32)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    # Rotation 0°/90°/180°/270° aléatoire
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

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

# ── Construction du modèle ─────────────────────────────────
print("\n[6/8] Construction du modèle EfficientNetB0...")

def build_transfer_model():
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
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="EfficientNetB0_HAM10000_v2")
    return model, base_model

model, base_model = build_transfer_model()
model.summary()

print(f"\n✓ Paramètres totaux       : {model.count_params():,}")
print(f"✓ Paramètres entraînables : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

def get_metrics():
    return [
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.Precision(name='precision'),
    ]

# ── PHASE 1 : Backbone gelé ────────────────────────────────
print("\n[7/8] Phase 1 — Entraînement (couches EfficientNet gelées)...")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=get_metrics()
)

callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "meilleur_modele_multiclass.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.CSVLogger(str(MODEL_DIR / "training_log_phase1.csv"))
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict,
    verbose=1
)

print(f"\n✓ Phase 1 terminée")
print(f"  Meilleure val_accuracy : {max(history1.history['val_accuracy'])*100:.2f}%")
print(f"  Meilleure val_auc      : {max(history1.history['val_auc'])*100:.2f}%")

# ── PHASE 2 : Fine-tuning ──────────────────────────────────
print(f"\n[8/8] Phase 2 — Fine-tuning ({UNFREEZE_LAYERS} couches dégelées)...")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# Garder BatchNorm gelées pour stabilité
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✓ Couches entraînables : {trainable_count:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=get_metrics()
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "meilleur_modele_multiclass.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
    ),
    keras.callbacks.CSVLogger(str(MODEL_DIR / "training_log_phase2.csv"))
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

# ── Évaluation finale ──────────────────────────────────────
print("\n── Évaluation finale sur le test set ──")
results = model.evaluate(test_ds, verbose=0)
metric_names = ['loss', 'accuracy', 'auc', 'recall', 'precision']
for name, val in zip(metric_names, results):
    if name == 'loss':
        print(f"✓ Test {name:12s} : {val:.4f}")
    else:
        print(f"✓ Test {name:12s} : {val*100:.2f}%")

y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n── Rapport de classification ──")
report = classification_report(
    y_true, y_pred,
    target_names=[CLASSES[c] for c in CLASS_NAMES],
    output_dict=True
)
print(classification_report(
    y_true, y_pred,
    target_names=[CLASSES[c] for c in CLASS_NAMES]
))

# ── Courbes combinées ──────────────────────────────────────
def combine_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

all_acc   = combine_histories(history1, history2, 'accuracy')
all_vacc  = combine_histories(history1, history2, 'val_accuracy')
all_loss  = combine_histories(history1, history2, 'loss')
all_vloss = combine_histories(history1, history2, 'val_loss')
all_auc   = combine_histories(history1, history2, 'auc')
all_vauc  = combine_histories(history1, history2, 'val_auc')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(all_acc) + 1)
phase2_start = len(history1.history['accuracy'])

for ax, train_vals, val_vals, title in zip(
    axes,
    [all_acc, all_loss, all_auc],
    [all_vacc, all_vloss, all_vauc],
    ['Accuracy', 'Loss', 'AUC']
):
    ax.plot(epochs_range, train_vals, label='Train', linewidth=2)
    ax.plot(epochs_range, val_vals,   label='Validation', linewidth=2)
    ax.axvline(x=phase2_start, color='red', linestyle='--', label='Fine-tuning', alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('EfficientNetB0 v2 — Transfer Learning + Fine-tuning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "courbes_entrainement_v2.png"), dpi=150)
plt.close()
print(f"\n✓ Courbes sauvegardées (accuracy + loss + AUC)")

# ── Matrice de confusion ───────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
labels = [CLASSES[c] for c in CLASS_NAMES]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('Matrice de Confusion — Valeurs absolues')
axes[0].set_ylabel('Vrai label')
axes[0].set_xlabel('Prédit')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('Matrice de Confusion — Normalisée (%)')
axes[1].set_ylabel('Vrai label')
axes[1].set_xlabel('Prédit')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(str(MODEL_DIR / "matrice_confusion_v2.png"), dpi=150)
plt.close()
print(f"✓ Matrices de confusion sauvegardées (absolue + normalisée)")

# ── Métriques par classe ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
classes_display = [CLASSES[c] for c in CLASS_NAMES]
f1_scores     = [report[c]['f1-score']  for c in classes_display]
recall_scores = [report[c]['recall']    for c in classes_display]
prec_scores   = [report[c]['precision'] for c in classes_display]

x = np.arange(len(classes_display))
width = 0.25
ax.bar(x - width, f1_scores,     width, label='F1-score',  color='steelblue')
ax.bar(x,         recall_scores, width, label='Recall',    color='coral')
ax.bar(x + width, prec_scores,   width, label='Precision', color='mediumseagreen')
ax.set_xticks(x)
ax.set_xticklabels(classes_display, rotation=30, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Métriques par classe — F1 / Recall / Precision')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "metriques_par_classe.png"), dpi=150)
plt.close()
print(f"✓ Graphique métriques par classe sauvegardé")

# ── Sauvegarde infos modèle ────────────────────────────────
test_acc  = results[1]
test_auc  = results[2]
test_rec  = results[3]
test_prec = results[4]

model_info = {
    "model_type"      : "EfficientNetB0_TransferLearning_v2",
    "version"         : "2.0",
    "classes"         : CLASS_NAMES,
    "class_names"     : CLASSES,
    "img_size"        : IMG_SIZE,
    "num_classes"     : NUM_CLASSES,
    "unfreeze_layers" : UNFREEZE_LAYERS,
    "test_accuracy"   : float(test_acc),
    "test_auc"        : float(test_auc),
    "test_recall"     : float(test_rec),
    "test_precision"  : float(test_prec),
}
with open(str(MODEL_DIR / "model_info_v2.json"), "w") as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("   ENTRAÎNEMENT v2 TERMINÉ !")
print(f"   Modèle    : models/meilleur_modele_multiclass.keras")
print(f"   Accuracy  : {test_acc*100:.2f}%")
print(f"   AUC       : {test_auc*100:.2f}%")
print(f"   Recall    : {test_rec*100:.2f}%")
print(f"   Precision : {test_prec*100:.2f}%")
print("=" * 60)
print("\nFichiers générés dans models/ :")
print("  courbes_entrainement_v2.png  (accuracy + loss + AUC)")
print("  matrice_confusion_v2.png     (absolue + normalisee)")
print("  metriques_par_classe.png     (F1 / Recall / Precision)")
print("  model_info_v2.json")
print("  training_log_phase1.csv")
print("  training_log_phase2.csv")