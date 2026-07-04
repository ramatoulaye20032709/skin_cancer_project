# ==============================================================
#   PROJET : Détection des tumeurs de la peau (ML/DL)
#   PHASE 4 : Application web avec Streamlit + Grad-CAM
#   Fichier : app_gradcam.py
#   Lancement : streamlit run app_gradcam.py
# ==============================================================

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import cv2

st.set_page_config(
    page_title="Détection des tumeurs cutanées",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR   = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
MODEL_PATH = BASE_DIR / "models" / "meilleur_modele.keras"
IMG_SIZE   = 224

@st.cache_resource
def charger_modele():
    if not MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model

def preparer_image(image):
    image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generer_gradcam(model, img_array, image_originale):
    try:
        derniere_conv = "conv2d_3"
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(derniere_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        img_resized   = np.array(image_originale.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
        superposition = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        return superposition, heatmap_resized
    except Exception as e:
        return None

def main():
    st.title("🔬 Détection des tumeurs de la peau")
    st.markdown("Système intelligent basé sur le **Deep Learning** pour l'analyse des lésions cutanées.")
    st.warning("⚠ Cet outil est une aide à la décision uniquement. Il ne remplace pas le diagnostic d'un dermatologue.")
    st.markdown("---")

    with st.spinner("Chargement du modèle..."):
        model = charger_modele()

    if model is None:
        st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
        st.stop()

    col_gauche, col_droite = st.columns([1, 1])

    with col_gauche:
        st.subheader("📁 Chargez une image")
        image_uploadee = st.file_uploader(
            "Formats acceptés : JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"]
        )
        if image_uploadee is not None:
            image = Image.open(image_uploadee)
            st.image(image, caption="Image uploadée", use_container_width=True)
            st.write(f"**Nom :** {image_uploadee.name}")
            st.write(f"**Taille :** {image.size[0]} × {image.size[1]} px")

    with col_droite:
        if image_uploadee is not None:
            st.subheader("📊 Analyse")
            if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    img_array  = preparer_image(image)
                    prediction = model.predict(img_array, verbose=0)
                    prob_malin = float(prediction[0][0])
                    prob_benin = 1 - prob_malin

                st.markdown("---")
                if prob_malin >= 0.5:
                    st.error("⚠ **MALIN** — Lésion potentiellement cancéreuse")
                else:
                    st.success("✓ **BÉNIN** — Lésion probablement non cancéreuse")

                st.markdown("#### Probabilités")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Bénin", f"{prob_benin*100:.1f}%")
                    st.progress(float(prob_benin))
                with c2:
                    st.metric("Malin", f"{prob_malin*100:.1f}%")
                    st.progress(float(prob_malin))

                confiance = max(prob_benin, prob_malin) * 100
                st.write(f"**Confiance du modèle :** {confiance:.1f}%")
                if confiance < 70:
                    st.warning("⚠ Confiance faible — Consultez un dermatologue.")

                st.markdown("---")
                st.subheader("🗺 Carte d'attention — Grad-CAM")
                st.caption("Les zones rouges sont celles qui ont le plus influencé la décision du modèle.")

                with st.spinner("Génération de la carte Grad-CAM..."):
                    resultat_gradcam = generer_gradcam(model, img_array, image)

                if resultat_gradcam is not None:
                    superposition, heatmap = resultat_gradcam
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(
                            np.array(image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))),
                            caption="Image originale",
                            use_container_width=True
                        )
                    with gc2:
                        st.image(
                            superposition,
                            caption="Zones analysées (rouge = important)",
                            use_container_width=True
                        )
                    st.caption("🔴 Rouge/Jaune = zones importantes  |  🔵 Bleu = zones peu importantes")
                else:
                    st.info("Grad-CAM non disponible pour ce modèle.")

                st.markdown("---")
                st.subheader("💡 Recommandation")
                if prob_malin >= 0.5:
                    st.markdown("""
- 🏥 **Consultez un dermatologue rapidement**
- 📋 Mentionnez cette analyse lors de votre consultation
- 🔍 Un examen dermatoscopique professionnel est recommandé
- ⏰ Ne tardez pas — la détection précoce est cruciale
                    """)
                else:
                    st.markdown("""
- ✅ La lésion semble bénigne selon le modèle
- 👁 Surveillez régulièrement vos grains de beauté
- 📅 Consultez un dermatologue pour un bilan annuel
- ⚠ En cas de changement, consultez immédiatement
                    """)

    st.markdown("---")
    st.markdown(
        "**Projet de fin d'études** — Licence Informatique 2025-2026  \n"
        "Modèle : CNN + Transfer Learning  |  "
        "Dataset : Melanoma Skin Cancer  |  "
        "Interprétabilité : Grad-CAM"
    )

if __name__ == "__main__":
    main()