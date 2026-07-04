# ==============================================================
#   MODELE DE VALIDATION — Dermatoscopique vs Non-Dermatoscopique
#   Utilise HAM10000 (positif) + ImageNet-mini (negatif)
#   Lancement : python train_validation.py
# ==============================================================

import os
import numpy as np
import shutil
import random
from pathlib import Path
from PIL import Image
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR      = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
HAM_DIR       = BASE_DIR / "data" / "ham10000" / "images"
IMAGENET_DIR  = BASE_DIR / "data" / "imagenet" / "imagenet-mini" / "train"
VAL_DIR       = BASE_DIR / "data" / "validation"
MODEL_DIR     = BASE_DIR / "models"
SKIN_DIR      = VAL_DIR / "skin"
NON_SKIN_DIR  = VAL_DIR / "non_skin"

IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 10
EPOCHS_FINETUNE = 10
NB_SKIN         = 2000
NB_NON_SKIN     = 2000

for d in [VAL_DIR, SKIN_DIR, NON_SKIN_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("   MODELE DE VALIDATION — Dermatoscopique vs Non-Dermatoscopique")
print("=" * 60)

# Etape 1 : Images dermatoscopiques
print("\n[1/5] Preparation des images dermatoscopiques...")
ham_images = list(HAM_DIR.glob("*.jpg"))
random.shuffle(ham_images)
ham_images = ham_images[:NB_SKIN]
for i, img_path in enumerate(ham_images):
    dst = SKIN_DIR / img_path.name
    if not dst.exists():
        shutil.copy(img_path, dst)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{NB_SKIN} images copiees")
print(f"✓ {len(ham_images)} images dermatoscopiques pretes")

# Etape 2 : Images non-dermatoscopiques
print("\n[2/5] Preparation des images non-dermatoscopiques...")
all_imagenet = []
for class_dir in IMAGENET_DIR.iterdir():
    if class_dir.is_dir():
        imgs = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        all_imagenet.extend(imgs)

print(f"  Total images ImageNet disponibles : {len(all_imagenet)}")
random.shuffle(all_imagenet)
selected = all_imagenet[:NB_NON_SKIN]

copied = 0
for i, img_path in enumerate(selected):
    dst = NON_SKIN_DIR / f"imagenet_{i:05d}.jpg"
    if not dst.exists():
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img.save(dst, 'JPEG', quality=90)
            copied += 1
        except:
            pass
    else:
        copied += 1
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{NB_NON_SKIN} images traitees")
print(f"✓ {copied} images non-dermatoscopiques pretes")

# Etape 3 : Dataset
print("\n[3/5] Preparation du dataset...")
skin_imgs     = list(SKIN_DIR.glob("*.jpg"))[:NB_SKIN]
non_skin_imgs = list(NON_SKIN_DIR.glob("*.jpg"))[:NB_NON_SKIN]
print(f"  Dermatoscopiques    : {len(skin_imgs)}")
print(f"  Non-dermatoscopiques: {len(non_skin_imgs)}")

filepaths = [str(p) for p in skin_imgs] + [str(p) for p in non_skin_imgs]
labels    = [1] * len(skin_imgs) + [0] * len(non_skin_imgs)

X_train, X_test, y_train, y_test = train_test_split(
    filepaths, labels, test_size=0.15, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label

def make_ds(paths, labels, aug=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if aug:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train, y_train, aug=True,  shuffle=True)
val_ds   = make_ds(X_val,   y_val,   aug=False, shuffle=False)
test_ds  = make_ds(X_test,  y_test,  aug=False, shuffle=False)

# Etape 4 : Modele
print("\n[4/5] Construction et entraînement du modele EfficientNetB0...")
print("=" * 60)

base = keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = keras.applications.efficientnet.preprocess_input(inputs)
x       = base(x, training=False)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(64, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_val = keras.Model(inputs, outputs, name="DermoValidation")

model_val.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks1 = [
    keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "modele_validation.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, verbose=1
    )
]

print("\nPhase 1 : Entraînement couches gelees...")
history1 = model_val.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FROZEN, callbacks=callbacks1, verbose=1
)

print("\nPhase 2 : Fine-tuning...")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_val.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks2 = [
    keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "modele_validation.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
    )
]

history2 = model_val.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FINETUNE, callbacks=callbacks2, verbose=1
)

# Etape 5 : Evaluation
print("\n[5/5] Evaluation finale...")
results = model_val.evaluate(test_ds, verbose=0)
test_acc = results[1]
test_auc = results[2]
print(f"✓ Test Accuracy : {test_acc*100:.2f}%")
print(f"✓ Test AUC      : {test_auc:.4f}")

y_true, y_pred_prob = [], []
for imgs, lbls in test_ds:
    preds = model_val.predict(imgs, verbose=0)
    y_true.extend(lbls.numpy())
    y_pred_prob.extend(preds.flatten())

y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
print("\n── Rapport de classification ──")
print(classification_report(
    y_true, y_pred,
    target_names=["Non-dermatoscopique", "Dermatoscopique"]
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-dermato", "Dermato"],
            yticklabels=["Non-dermato", "Dermato"])
plt.title('Matrice de Confusion - Modele Validation')
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "validation_confusion.png"), dpi=150)
plt.close()

print("\n" + "=" * 60)
print("   MODELE DE VALIDATION CREE !")
print(f"   Sauvegarde : models/modele_validation.keras")
print(f"   Accuracy   : {test_acc*100:.2f}%")
print(f"   AUC        : {test_auc:.4f}")
print("=" * 60)
