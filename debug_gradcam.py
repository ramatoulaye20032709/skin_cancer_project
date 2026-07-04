import keras
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

MODEL_PATH = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project\models\meilleur_modele_multiclass.keras")
model = keras.models.load_model(str(MODEL_PATH), compile=False)

# Forcer la construction
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy)
print("✓ Modèle chargé")

# Créer un modèle intermédiaire avec top_conv comme sortie
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer("top_conv").output, model.output]
)
print("✓ grad_model créé")

# Image test
img_dir = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project\data\ham10000\images")
img_path = str(list(img_dir.glob("*.jpg"))[0])
image = Image.open(img_path).convert('RGB')
img_array = tf.cast(np.expand_dims(np.array(image.resize((224, 224))), axis=0), tf.float32)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
print(f"✓ conv_out shape : {conv_outputs.shape}")
print(f"✓ grads shape    : {grads.shape}")
print("✓ Grad-CAM fonctionne !")