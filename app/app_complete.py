# ══════════════════════════════════════════════════════════
#  PAGE 1 — ANALYSE (CORRIGÉ)
# ══════════════════════════════════════════════════════════

import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime

# sécurité page (IMPORTANT STREAMLIT)
if "page" not in st.session_state:
    st.session_state.page = "🔬 Analyse"

page = st.session_state.page

if page == "🔬 Analyse":

    st.title("🔬 Analyse de lésion cutanée")
    st.warning("Cet outil est une aide à la décision uniquement.")

    # =========================
    # sécurité variables
    # =========================
    if "historique" not in st.session_state:
        st.session_state.historique = []

    # =========================
    # vérification modèles
    # =========================
    if model is None:
        st.error("Modèle introuvable")
        st.stop()

    if model_val is not None:
        st.success("✓ Modèle de validation actif")

    # =========================
    # infos patient
    # =========================
    with st.expander("👤 Informations du patient", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            nom_patient = st.text_input("Nom", value="Patient_001")
        with c2:
            age_patient = st.number_input("Age", 1, 120, 45)
        with c3:
            sexe_patient = st.selectbox("Sexe", ["Non précisé", "Homme", "Femme"])
        with c4:
            localisation = st.selectbox(
                "Localisation",
                ["Visage", "Cou", "Torse", "Dos", "Bras", "Jambe", "Pied", "Main"]
            )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =========================
    # UPLOAD MULTI IMAGES
    # =========================
    with col1:
        images_upload = st.file_uploader(
            "📁 Charger plusieurs images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if images_upload:
            st.success(f"{len(images_upload)} image(s) chargée(s)")

    # =========================
    # RESULTATS
    # =========================
    with col2:
        st.subheader("📊 Résultats")

        if images_upload and len(images_upload) > 0:

            if st.button("🔍 Lancer l'analyse"):

                for img_file in images_upload:

                    st.markdown("---")
                    st.subheader(f"📷 {img_file.name}")

                    image = Image.open(img_file)
                    st.image(image, width=250)

                    img_array = preparer_image(image)

                    # prediction
                    prediction = model.predict(img_array, verbose=0)[0]
                    class_idx = int(np.argmax(prediction))

                    classe_pred = CLASS_NAMES[class_idx]
                    classe_nom = CLASSES[classe_pred]

                    confiance = float(prediction[class_idx]) * 100
                    niveau_risque, icone, couleur = get_niveau_risque(classe_pred)

                    # affichage résultat
                    if niveau_risque == "MALIN":
                        st.error(f"🔴 {classe_nom}")
                    elif niveau_risque == "SUSPECT":
                        st.warning(f"⚠️ {classe_nom}")
                    else:
                        st.success(f"✅ {classe_nom}")

                    st.info(f"Confiance : {confiance:.1f}%")

                    # historique SAFE
                    st.session_state.historique.append({
                        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "patient": nom_patient,
                        "age": age_patient,
                        "diagnostic": classe_nom,
                        "niveau": niveau_risque,
                        "confiance": f"{confiance:.1f}%",
                        "fichier": img_file.name,
                    })

                st.success("✅ Analyse terminée")

        else:
            st.info("Charge au moins une image")