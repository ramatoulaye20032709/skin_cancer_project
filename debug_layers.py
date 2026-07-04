import keras
from pathlib import Path

MODEL_PATH = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project\models\meilleur_modele_multiclass.keras")
model = keras.models.load_model(str(MODEL_PATH), compile=False)

print("=== COUCHES PRINCIPALES ===")
for i, layer in enumerate(model.layers):
    print(f"{i:>3} | {layer.name:50s} | {type(layer).__name__}")
    if hasattr(layer, 'layers'):
        print(f"      └─ Sous-modèle avec {len(layer.layers)} couches")
        for j, sublayer in enumerate(layer.layers[-5:]):  # 5 dernières
            print(f"         {j} | {sublayer.name:45s} | {type(sublayer).__name__}")