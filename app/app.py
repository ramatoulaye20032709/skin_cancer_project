# ==============================================================
#   PROJET : DermIA — Détection des tumeurs de la peau (ML/DL)
#   Version 4 : + Métriques de performance + Upload multi-images
#               (batch) + Explication de la confiance + Bouton Détails
#   Fichier : app.py
#   Lancement : streamlit run app.py
# ==============================================================

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
import tensorflow as tf
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Configuration de la page ───────────────────────────────
st.set_page_config(
    page_title="DermIA — Détection des tumeurs cutanées",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé — design médical professionnel ────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ink: #16302A;
        --ink-soft: #5B7268;
        --brand-dark: #0A4D3C;
        --brand: #1A7A5E;
        --brand-light: #2EAA84;
        --bg: #F6FBF9;
        --surface: #FFFFFF;
        --border: #E3F1EC;
        --danger: #C0392B;
        --warning: #C8860D;
        --shadow-sm: 0 1px 3px rgba(10,77,60,0.07);
        --shadow-md: 0 6px 20px rgba(10,77,60,0.10);
        --shadow-lg: 0 12px 32px rgba(10,77,60,0.14);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: var(--bg); }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Sora', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }
    p, span, label, li, div { color: var(--ink); }
    ::selection { background: var(--brand-light); color: white; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #C7E6DA; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--brand-light); }

    /* ── En-tête ── */
    .dermia-header {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, var(--brand-dark) 0%, var(--brand) 100%);
        padding: 32px 36px;
        border-radius: 16px;
        color: white;
        margin-bottom: 26px;
        box-shadow: var(--shadow-lg);
    }
    .dermia-header::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .dermia-header h1 {
        margin: 0; font-size: 34px; font-weight: 800; color: white !important;
    }
    .dermia-header p { margin: 6px 0 0 0; opacity: 0.92; font-size: 15px; font-weight: 500; }

    /* ── Cartes ── */
    .dermia-card {
        background: var(--surface);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        margin-bottom: 14px;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .dermia-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }

    .class-chip {
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
        min-height: 130px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        box-shadow: var(--shadow-sm);
    }
    .class-chip:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }

    .abcde-chip {
        background-color: #F0FAF6;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        min-height: 110px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .abcde-chip:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }

    .quality-ok { color: var(--brand); font-weight: 700; }
    .quality-bad { color: var(--danger); font-weight: 700; }

    /* ── Boutons ── */
    .stButton > button {
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.55em 1.2em;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        box-shadow: var(--shadow-sm);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--brand-dark), var(--brand));
        color: white;
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background: var(--surface);
        color: var(--brand-dark);
        border: 1.5px solid var(--border);
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow-md); filter: brightness(1.04); }
    .stButton > button:active { transform: translateY(0); filter: brightness(0.97); }
    .stButton > button:focus-visible { outline: 2px solid var(--brand-light); outline-offset: 2px; }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: var(--surface);
        color: var(--brand-dark);
        border: 1.5px solid var(--brand-light);
        transition: all 0.15s ease;
    }
    .stDownloadButton > button:hover {
        background: var(--brand-light);
        color: white;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    /* ── Onglets (style pilule) ── */
    [data-baseweb="tab-list"] {
        gap: 6px;
        background: #EFF8F4;
        padding: 6px;
        border-radius: 12px;
    }
    [data-baseweb="tab"] {
        border-radius: 9px !important;
        font-weight: 600;
        color: var(--ink-soft) !important;
        transition: all 0.18s ease;
    }
    [data-baseweb="tab"]:hover { background: rgba(26,122,94,0.08); }
    [data-baseweb="tab"][aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--brand-dark) !important;
        box-shadow: var(--shadow-sm);
    }
    [data-baseweb="tab-highlight"] { background: transparent !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-sm);
        margin-bottom: 10px;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--brand-dark);
    }
    [data-testid="stExpander"]:hover { box-shadow: var(--shadow-md); }

    /* ── Métriques ── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--brand);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.18s ease;
    }
    [data-testid="stMetric"]:hover { box-shadow: var(--shadow-md); }
    [data-testid="stMetricValue"] { font-family: 'Sora', sans-serif; color: var(--brand-dark); }

    /* ── Alertes ── */
    .stAlert {
        border-radius: 12px !important;
        box-shadow: var(--shadow-sm);
    }

    /* ── Téléchargement de fichier ── */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1.5px dashed #B9DECF !important;
        background: #FAFFFD;
        transition: border-color 0.18s ease, background 0.18s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--brand-light) !important;
        background: #F0FAF6;
    }

    /* ── Barre de progression ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--brand-dark), var(--brand-light)) !important;
        border-radius: 8px;
    }

    /* ── Tableaux ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* ── Barre latérale ── */
    section[data-testid="stSidebar"] {
        background-color: #F0FAF6;
        border-right: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# ── Chemins ────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\ramatoulaye sy\skin_cancer_project")
MODEL_PATH   = BASE_DIR / "models" / "meilleur_modele_v3.keras"
VALIDATION_MODEL_PATH = BASE_DIR / "models" / "modele_validation.keras"
HISTORY_PATH = BASE_DIR / "data" / "historique_patients.csv"
METRICS_PATH = BASE_DIR / "data" / "metriques_modele.json"
CM_IMAGE_PATH  = BASE_DIR / "data" / "confusion_matrix.png"
ROC_IMAGE_PATH = BASE_DIR / "data" / "roc_curves.png"
IMG_SIZE     = 224

# ── Classes HAM10000 ───────────────────────────────────────
CLASSES = {
    0: ('Melanoma',               'MEL',   'MALIN',        '🔴', '#C0392B'),
    1: ('Nevus',                  'NV',    'BÉNIN',        '🟢', '#1A7A5E'),
    2: ('Basal Cell Carcinoma',   'BCC',   'MALIN',        '🔴', '#E67E22'),
    3: ('Benign Keratosis',       'BKL',   'BÉNIN',        '🟢', '#2EAA84'),
    4: ('Dermatofibroma',         'DF',    'BÉNIN',        '🟢', '#6B7F78'),
    5: ('Vascular Lesion',        'VASC',  'BÉNIN',        '🟢', '#9B59B6'),
    6: ('Actinic Keratosis',      'AKIEC', 'PRÉCANCÉREUX', '🟡', '#F1C40F'),
}
CODES_ORDRE = [CLASSES[i][1] for i in range(len(CLASSES))]

CLASSES_INFO = {
    'MEL':   "Mélanome — cancer issu des mélanocytes. 11,1% du dataset mais responsable de ~75% des décès par cancer cutané. Détection précoce cruciale.",
    'NV':    "Nævus mélanocytaire (grain de beauté) — lésion bénigne très fréquente (66,9% du dataset). Souvent confondu avec le mélanome.",
    'BCC':   "Carcinome basocellulaire — cancer le plus fréquent chez l'humain (~80% des cancers cutanés). Rarement fatal, peu métastatique.",
    'BKL':   "Kératose bénigne — regroupe kératoses séborrhéiques et lentigos solaires. Lésion bénigne liée au vieillissement.",
    'DF':    "Dermatofibrome — tumeur bénigne du derme, petite et ferme. Classe rare dans HAM10000 (1,1%).",
    'VASC':  "Lésion vasculaire — angiomes, angiokeratomes. Généralement bénigne, classe rare (1,4%).",
    'AKIEC': "Kératose actinique — lésion précancéreuse liée aux UV. Peut évoluer en carcinome épidermoïde (5-10% des cas non traités).",
}

DISTRIBUTION = [11.1, 66.9, 5.1, 11.0, 1.1, 1.4, 3.3]

# ── Persistance historique (CSV local) ─────────────────────
def charger_historique():
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH).to_dict('records')
        except Exception:
            return []
    return []

def sauvegarder_historique(historique):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(historique).to_csv(HISTORY_PATH, index=False)

# ── Initialisation session state ───────────────────────────
if 'historique' not in st.session_state:
    st.session_state.historique = charger_historique()
if 'afficher_details' not in st.session_state:
    st.session_state.afficher_details = {}
if 'batch_resultats' not in st.session_state:
    st.session_state.batch_resultats = None
if 'gradcam_cache' not in st.session_state:
    st.session_state.gradcam_cache = {}
if 'pdf_cache' not in st.session_state:
    st.session_state.pdf_cache = {}

# ── Chargement du modèle ───────────────────────────────────
@st.cache_resource
def charger_modele():
    if not MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model

# ── Chargement du modèle de validation peau / pas-peau ──────
@st.cache_resource
def charger_modele_validation():
    if not VALIDATION_MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(str(VALIDATION_MODEL_PATH))
    return model

# Seuil de décision sur le score sigmoïde (1 = peau/dermatoscopique, 0 = autre).
# À ajuster si besoin selon la matrice de confusion obtenue à l'entraînement
# (train_validation.py) pour équilibrer faux rejets / faux positifs.
SEUIL_PEAU = 0.5

# ── Prétraitement ──────────────────────────────────────────
def preparer_image(image):
    image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Validation qualité de l'image ──────────────────────────
def valider_qualite_image(pil_image):
    """
    Analyse 4 critères de qualité avant classification :
    netteté, résolution, exposition, contraste.
    """
    img_rgb = np.array(pil_image.convert('RGB'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape

    resultats = {}

    resultats['resolution'] = {
        'valeur': f"{w} × {h} px",
        'ok': (w >= 100 and h >= 100)
    }

    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    resultats['nettete'] = {
        'valeur': f"{laplacian_var:.1f}",
        'ok': laplacian_var > 80
    }

    pct_sature = np.mean(img_gray >= 250) * 100
    pct_noir   = np.mean(img_gray <= 5) * 100
    resultats['exposition'] = {
        'valeur': f"{pct_sature:.1f}% sur-exposé / {pct_noir:.1f}% sous-exposé",
        'ok': (pct_sature < 5 and pct_noir < 5)
    }

    contraste = img_gray.std()
    resultats['contraste'] = {
        'valeur': f"{contraste:.1f} / 255",
        'ok': contraste > 10
    }

    qualite_globale = all(r['ok'] for r in resultats.values())
    return resultats, qualite_globale

# ── Grad-CAM ───────────────────────────────────────────────
def generate_gradcam(model, img_array, layer_name='top_conv'):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index    = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads        = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0)
        heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        img_orig        = img_array[0].astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed    = cv2.addWeighted(img_orig, 0.6, heatmap_colored, 0.4, 0)
        return img_orig, heatmap_colored, superimposed
    except Exception as e:
        st.warning(f"Grad-CAM non disponible : {e}")
        return None, None, None

# ── Génération du rapport PDF ──────────────────────────────
def generer_pdf(patient_nom, patient_age, nom_classe, code_classe, statut,
                 confidence, predictions, img_orig_arr, heatmap_arr, superimposed_arr):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=1.5*cm, bottomMargin=1.5*cm,
                             leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleDermIA', parent=styles['Title'],
                                  textColor=colors.HexColor('#0A4D3C'), fontSize=24)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     textColor=colors.HexColor('#1A7A5E'), fontSize=12, spaceAfter=12)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    textColor=colors.HexColor('#0A4D3C'), fontSize=14, spaceBefore=12, spaceAfter=6)
    normal_style = styles['Normal']
    alert_style = ParagraphStyle('Alert', parent=styles['Normal'],
                                  textColor=colors.HexColor('#C0392B'), fontSize=10, spaceAfter=6)

    story = []
    story.append(Paragraph("DermIA", title_style))
    story.append(Paragraph("Rapport d'analyse dermoscopique automatisée (ML/DL)", subtitle_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Ce rapport est généré par un outil d'aide à la décision académique. "
        "Il ne remplace pas le diagnostic d'un dermatologue certifié.",
        alert_style
    ))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Informations patient", section_style))
    info_data = [
        ["Nom du patient", patient_nom or "Non renseigné"],
        ["Âge", f"{patient_age} ans" if patient_age else "Non renseigné"],
        ["Date de l'analyse", datetime.now().strftime("%d/%m/%Y a %H:%M")],
        ["N. de session", str(len(st.session_state.historique))],
    ]
    info_table = Table(info_data, colWidths=[6*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5F0')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0A4D3C')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0EDE5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Résultat de la classification", section_style))
    statut_color = colors.HexColor('#C0392B') if statut == 'MALIN' else (
        colors.HexColor('#F1C40F') if statut == 'PRÉCANCÉREUX' else colors.HexColor('#1A7A5E'))
    result_style = ParagraphStyle('Result', parent=styles['Normal'],
                                   fontSize=14, textColor=statut_color, fontName='Helvetica-Bold')
    story.append(Paragraph(f"{statut} — {nom_classe} ({code_classe})", result_style))
    story.append(Paragraph(f"Confiance du modèle : {confidence:.1f}%", normal_style))
    story.append(Paragraph(CLASSES_INFO.get(code_classe, ""), normal_style))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Probabilités par classe", section_style))
    prob_data = [["Classe", "Probabilité"]]
    for i, prob in enumerate(predictions):
        nom, code, _, _, _ = CLASSES[i]
        prob_data.append([f"{nom} ({code})", f"{prob*100:.2f}%"])
    prob_table = Table(prob_data, colWidths=[10*cm, 5*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0A4D3C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0EDE5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0FAF6')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(PageBreak())

    story.append(Paragraph("Visualisation et interprétabilité (Grad-CAM)", section_style))
    story.append(Paragraph(
        "La carte Grad-CAM met en évidence les zones de l'image ayant le plus "
        "influencé la décision du modèle (zones rouges = forte influence).",
        normal_style
    ))
    story.append(Spacer(1, 0.3*cm))

    def array_to_rlimage(arr, width=5.5*cm):
        img_pil = Image.fromarray(arr.astype('uint8'))
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return RLImage(img_buffer, width=width, height=width)

    if img_orig_arr is not None:
        img_row = [[
            array_to_rlimage(img_orig_arr),
            array_to_rlimage(heatmap_arr),
            array_to_rlimage(superimposed_arr)
        ]]
        story.append(Table(img_row, colWidths=[5.5*cm, 5.5*cm, 5.5*cm]))
        cap_table = Table([["Image originale", "Carte Grad-CAM", "Superposition"]],
                           colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
        cap_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#6B7F78')),
        ]))
        story.append(cap_table)

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Rappel clinique — Règle ABCDE", section_style))
    abcde_data = [
        ["A", "Asymétrie", "Forme irrégulière, sans axe de symétrie"],
        ["B", "Bords", "Contours flous, dentelés ou mal délimités"],
        ["C", "Couleur", "Présence de plusieurs teintes (brun, noir, rouge...)"],
        ["D", "Diamètre", "Supérieur à 6 mm"],
        ["E", "Évolution", "Changement récent de taille, forme ou couleur"],
    ]
    abcde_table = Table(abcde_data, colWidths=[1*cm, 3*cm, 11*cm])
    abcde_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0A4D3C')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('FONTNAME', (0, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0EDE5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(abcde_table)
    story.append(Spacer(1, 0.6*cm))

    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.HexColor('#6B7F78'))
    story.append(Paragraph(
        "DermIA — Modele EfficientNetB0 + Transfer Learning | Dataset HAM10000 (Tschandl et al., 2018) | "
        "Interpretabilite Grad-CAM (Selvaraju et al., 2017) | Auteur : Ramatoulaye Sy - 2026",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── NOUVEAU : indicateurs de confiance ──────────────────────
def calculer_entropie(predictions):
    """Entropie de Shannon normalisée entre 0 (certitude totale) et 1 (incertitude max)."""
    p = np.clip(predictions, 1e-12, 1.0)
    entropie = -np.sum(p * np.log2(p))
    entropie_max = np.log2(len(predictions))
    return float(entropie / entropie_max)

def calculer_marge_confiance(predictions):
    """Écart en points de % entre la classe prédite et la 2e classe la plus probable."""
    tries = np.sort(predictions)[::-1]
    return float((tries[0] - tries[1]) * 100)

# ── NOUVEAU : explication de la confiance (pour la soutenance) ─
def afficher_explication_confiance():
    with st.expander("ℹ️ Comment la confiance du modèle est-elle calculée ?"):
        st.markdown("""
        Le réseau DermIA se termine par une couche **softmax** qui transforme les
        scores bruts en probabilités pour chacune des 7 classes — leur somme vaut
        toujours 100%.

        La **confiance affichée** correspond simplement à la probabilité de la
        classe retenue (la plus élevée parmi les 7).
        """)
        st.latex(r"\text{Confiance} = \max_i \; \text{softmax}(z)_i \times 100")
        st.markdown("""
        **À retenir pour la soutenance :**
        - Une confiance élevée traduit la certitude *du modèle*, pas une vérité
          absolue — le modèle peut être confiant et se tromper.
        - Deux indicateurs complémentaires sont calculés dans la section
          **« Détails »** de chaque résultat :
            - la **marge de confiance** (écart avec la 2ᵉ classe la plus probable),
            - l'**entropie** (incertitude globale sur les 7 classes) —
              utile pour repérer les cas où le modèle hésite, par exemple entre
              Mélanome et Nevus, deux classes visuellement proches.
        """)

# ── NOUVEAU : contenu du bouton "Détails" ──────────────────
def afficher_details_prediction(predictions, resultats_qualite):
    st.markdown("##### 🔍 Détails de la prédiction")

    marge = calculer_marge_confiance(predictions)
    entropie = calculer_entropie(predictions)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Marge avec la 2ᵉ classe", f"{marge:.1f} pts")
        if marge < 10:
            st.caption("⚠️ Marge faible : le modèle hésite entre deux classes proches.")
        else:
            st.caption("✓ Marge confortable entre les deux classes les plus probables.")
    with col2:
        st.metric("Entropie (incertitude)", f"{entropie*100:.1f}%")
        if entropie > 0.5:
            st.caption("⚠️ Distribution incertaine sur plusieurs classes.")
        else:
            st.caption("✓ Le modèle est concentré sur une classe dominante.")

    st.markdown("**Probabilités complètes, triées :**")
    lignes = []
    ordre = np.argsort(predictions)[::-1]
    for rang, i in enumerate(ordre, start=1):
        nom, code, statut, emoji, _ = CLASSES[i]
        lignes.append({
            "Rang": rang,
            "Classe": f"{emoji} {code} — {nom}",
            "Statut": statut,
            "Probabilité": f"{predictions[i]*100:.2f}%"
        })
    st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)

    st.markdown("**Rappel — qualité de l'image analysée :**")
    labels = {'resolution': 'Résolution', 'nettete': 'Netteté',
              'exposition': 'Exposition', 'contraste': 'Contraste'}
    lignes_q = []
    for key, label in labels.items():
        r = resultats_qualite[key]
        lignes_q.append({"Critère": label, "Valeur": r['valeur'],
                          "Conforme": "✓" if r['ok'] else "✗"})
    st.dataframe(pd.DataFrame(lignes_q), use_container_width=True, hide_index=True)

# ── Diagramme statique des 7 classes ────────────────────────
def afficher_diagramme_7_classes():
    st.markdown("#### 🧬 Les 7 classes de lésions cutanées (HAM10000)")
    cols = st.columns(7)
    for i, col in enumerate(cols):
        nom, code, statut, emoji, color = CLASSES[i]
        with col:
            st.markdown(
                f"""<div class="class-chip" style="background-color:{color}1A; border-left:4px solid {color};">
                <div style="font-size:20px;">{emoji}</div>
                <div style="font-weight:bold; color:{color}; font-size:13px;">{code}</div>
                <div style="font-size:10px; color:#555;">{nom}</div>
                <div style="font-size:11px; font-weight:bold; margin-top:4px;">{DISTRIBUTION[i]}%</div>
                <div style="font-size:9px; color:{color};">{statut}</div>
                </div>""",
                unsafe_allow_html=True
            )

# ── Règle ABCDE ──────────────────────────────────────────────
def afficher_regle_abcde():
    st.markdown("#### 📋 Rappel clinique — Règle ABCDE")
    abcde = [
        ("A", "Asymétrie", "Forme irrégulière, sans axe de symétrie"),
        ("B", "Bords", "Contours flous, dentelés ou mal délimités"),
        ("C", "Couleur", "Plusieurs teintes (brun, noir, rouge, bleu...)"),
        ("D", "Diamètre", "Supérieur à 6 mm"),
        ("E", "Évolution", "Changement récent de taille, forme ou couleur"),
    ]
    cols = st.columns(5)
    for col, (letter, titre, desc) in zip(cols, abcde):
        with col:
            st.markdown(
                f"""<div class="abcde-chip">
                <div style="font-size:22px; font-weight:bold; color:#1A7A5E;">{letter}</div>
                <div style="font-size:12px; font-weight:bold; color:#0A4D3C;">{titre}</div>
                <div style="font-size:10px; color:#555; margin-top:4px;">{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )

# ── Affichage des résultats de validation qualité ───────────
def afficher_validation_qualite(resultats, qualite_globale):
    st.markdown("#### ✅ Validation automatique de la qualité de l'image")
    cols = st.columns(4)
    labels = {
        'resolution': '📐 Résolution',
        'nettete':    '🔍 Netteté',
        'exposition': '💡 Exposition',
        'contraste':  '🎨 Contraste'
    }
    for col, (key, label) in zip(cols, labels.items()):
        r = resultats[key]
        css_class = "quality-ok" if r['ok'] else "quality-bad"
        icone = "✓" if r['ok'] else "✗"
        with col:
            st.markdown(
                f"""<div class="dermia-card" style="text-align:center;">
                <div style="font-size:13px; color:#555;">{label}</div>
                <div class="{css_class}" style="font-size:18px;">{icone}</div>
                <div style="font-size:11px; color:#888;">{r['valeur']}</div>
                </div>""",
                unsafe_allow_html=True
            )

    if qualite_globale:
        st.success("✓ Image de qualité suffisante pour l'analyse.")
    else:
        st.warning(
            "⚠️ Cette image présente des problèmes de qualité (flou, exposition, contraste...). "
            "L'analyse reste possible mais la fiabilité du résultat peut être réduite."
        )
    return qualite_globale

# ── NOUVEAU : chargement + affichage des métriques de performance ─
def charger_metriques_modele():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def afficher_performance_modele():
    st.subheader("📊 Performance du modèle DermIA")
    st.markdown(
        "Ces métriques sont calculées sur un **jeu de test indépendant** "
        "(images jamais vues pendant l'entraînement), conformément aux "
        "bonnes pratiques d'évaluation en apprentissage automatique."
    )

    metriques = charger_metriques_modele()

    if metriques is None:
        st.warning(
            "⚠️ Aucun fichier de métriques trouvé. Lance d'abord le script "
            "**`evaluer_performance.py`** (fourni à côté de `app.py`) pour "
            "générer `metriques_modele.json` et les graphiques associés à "
            "partir de ton jeu de test."
        )
        st.code("python evaluer_performance.py", language="bash")
        st.caption(f"Le fichier est attendu ici : {METRICS_PATH}")
        return

    st.caption(
        f"Évalué sur {metriques.get('nombre_images_test', '?')} images de test "
        f"— {str(metriques.get('date_evaluation', ''))[:10]}"
    )

    rapport = metriques.get("rapport_par_classe", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy globale", f"{metriques.get('accuracy_globale', 0)*100:.1f}%")
    with col2:
        st.metric("📐 Précision (macro)", f"{rapport.get('macro avg', {}).get('precision', 0)*100:.1f}%")
    with col3:
        st.metric("🔁 Rappel (macro)", f"{rapport.get('macro avg', {}).get('recall', 0)*100:.1f}%")
    with col4:
        st.metric("⚖️ F1-score (macro)", f"{rapport.get('macro avg', {}).get('f1-score', 0)*100:.1f}%")

    if metriques.get("auc_roc_macro"):
        st.metric("📈 AUC-ROC (macro, one-vs-rest)", f"{metriques['auc_roc_macro']*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Détail par classe")
    lignes = []
    for code in CODES_ORDRE:
        r = rapport.get(code)
        if not r:
            continue
        auc_classe = metriques.get("auc_roc_par_classe", {}).get(code)
        lignes.append({
            "Classe": code,
            "Précision": f"{r['precision']*100:.1f}%",
            "Rappel": f"{r['recall']*100:.1f}%",
            "F1-score": f"{r['f1-score']*100:.1f}%",
            "Support (images test)": int(r["support"]),
            "AUC-ROC": f"{auc_classe*100:.1f}%" if auc_classe is not None else "N/A",
        })
    if lignes:
        st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matrice de confusion")
        if CM_IMAGE_PATH.exists():
            st.image(str(CM_IMAGE_PATH), width="stretch")
        else:
            st.info("Image non trouvée — relance evaluer_performance.py.")
    with col2:
        st.markdown("#### Courbes ROC")
        if ROC_IMAGE_PATH.exists():
            st.image(str(ROC_IMAGE_PATH), width="stretch")
        else:
            st.info("Image non trouvée — relance evaluer_performance.py.")

    with st.expander("ℹ️ Comment interpréter ces métriques ? (utile pour le jury)"):
        st.markdown("""
        - **Précision** : parmi les lésions prédites comme appartenant à une
          classe, quelle proportion l'est réellement.
        - **Rappel (sensibilité)** : parmi les lésions réellement de cette
          classe, quelle proportion a été correctement détectée — crucial
          pour le mélanome, où un rappel faible signifie des cas manqués.
        - **F1-score** : moyenne harmonique précision/rappel, particulièrement
          pertinente ici car les classes sont déséquilibrées
          (66,9% de Nevus contre 1,1% de Dermatofibrome).
        - **AUC-ROC** : capacité du modèle à séparer une classe des autres,
          indépendamment du seuil de décision choisi.
        - **Matrice de confusion** : montre précisément quelles classes sont
          confondues entre elles (ex : Mélanome vs Nevus, souvent proches
          visuellement).
        """)

# ── Sidebar : formulaire patient + recherche ────────────────
def sidebar_patient():
    st.sidebar.markdown("## 🧑‍⚕️ Dossier patient")
    patient_nom = st.sidebar.text_input("Nom du patient", placeholder="Ex : Aïssatou Diop")
    patient_age = st.sidebar.number_input("Âge", min_value=0, max_value=120, value=30, step=1)

    st.sidebar.markdown("---")
    st.sidebar.metric("👥 Total patients analysés", len(st.session_state.historique))

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔎 Recherche dans l'historique")
    recherche = st.sidebar.text_input("Rechercher un patient par nom")

    if recherche and st.session_state.historique:
        df_hist = pd.DataFrame(st.session_state.historique)
        resultats_recherche = df_hist[
            df_hist['nom'].str.contains(recherche, case=False, na=False)
        ]
        if not resultats_recherche.empty:
            st.sidebar.success(f"{len(resultats_recherche)} résultat(s) trouvé(s)")
            st.sidebar.dataframe(resultats_recherche, use_container_width=True, hide_index=True)
        else:
            st.sidebar.info("Aucun patient trouvé avec ce nom.")

    return patient_nom, patient_age

# ── Statistiques globales (sidebar) ─────────────────────────
def sidebar_statistiques():
    if not st.session_state.historique:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Statistiques globales")
    df_hist = pd.DataFrame(st.session_state.historique)
    if 'classe' in df_hist.columns:
        repartition = df_hist['classe'].value_counts()
        st.sidebar.bar_chart(repartition)

# ── NOUVEAU : affichage du résultat pour UNE image du lot ───
def afficher_resultat_image(idx, resultat, model):
    """Affiche le résultat déjà calculé (au moment du clic sur "Analyser") pour
    une image. La prédiction et le contrôle qualité sont déjà prêts (calculés en
    lot). Le Grad-CAM et le PDF, plus coûteux, ne sont calculés qu'à la demande
    (bouton) et mis en cache -> essentiel pour rester fluide avec 50 images."""
    nom_fichier   = resultat["nom_fichier"]
    image         = resultat["image_pil"]
    img_array     = resultat["img_array"]
    predictions   = resultat["predictions"]
    confidence    = resultat["confidence"]
    nom_classe    = resultat["nom_classe"]
    code_classe   = resultat["code_classe"]
    statut        = resultat["statut"]
    emoji         = resultat["emoji"]
    resultats_qualite = resultat["resultats_qualite"]
    qualite_ok    = resultat["qualite_ok"]
    score_peau    = resultat.get("score_peau")
    est_peau      = resultat.get("est_peau", True)
    cle = f"{idx}_{nom_fichier}"

    if not est_peau:
        st.image(image, caption=nom_fichier, width="stretch")
        st.error(
            "⚠️ Cette image ne ressemble pas à une lésion cutanée "
            f"(score de détection peau : {score_peau*100:.1f}%, seuil requis : {SEUIL_PEAU*100:.0f}%).\n\n"
            "DermIA est conçu uniquement pour l'analyse de lésions cutanées. "
            "Aucun diagnostic n'est proposé pour cette image, et elle n'a pas été "
            "ajoutée à l'historique patient. Vérifiez que la bonne image a été soumise."
        )
        cle_force = f"force_{cle}"
        if cle_force not in st.session_state.afficher_details:
            st.session_state.afficher_details[cle_force] = False
        if st.button("👁️ Afficher quand même le résultat du modèle de diagnostic (non recommandé)",
                      key=f"btn_{cle_force}"):
            st.session_state.afficher_details[cle_force] = not st.session_state.afficher_details[cle_force]
            st.rerun()
        if not st.session_state.afficher_details[cle_force]:
            return
        st.caption("Résultat affiché à titre purement indicatif — non fiable, l'image n'étant pas reconnue comme une lésion cutanée :")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=nom_fichier, width="stretch")
    with col2:
        st.write(f"**Fichier :** {nom_fichier}")
        st.write(f"**Taille :** {image.size[0]} × {image.size[1]} px")
        if statut == 'MALIN':
            st.error(f"{emoji} **{statut}** — {nom_classe} ({code_classe}) — {confidence:.1f}%")
        elif statut == 'PRÉCANCÉREUX':
            st.warning(f"{emoji} **{statut}** — {nom_classe} ({code_classe}) — {confidence:.1f}%")
        else:
            st.success(f"{emoji} **{statut}** — {nom_classe} ({code_classe}) — {confidence:.1f}%")
        st.caption(CLASSES_INFO.get(code_classe, ""))

    afficher_validation_qualite(resultats_qualite, qualite_ok)

    st.markdown("**Probabilités par classe :**")
    chart_data = {f"{CLASSES[i][1]} - {CLASSES[i][0]}": float(p) for i, p in enumerate(predictions)}
    st.bar_chart(chart_data)

    deja_ouvert = st.session_state.afficher_details.get(cle, False)
    label_bouton = "🔼 Masquer les détails" if deja_ouvert else "🔍 Voir les détails"
    if st.button(label_bouton, key=f"btn_details_{cle}"):
        st.session_state.afficher_details[cle] = not deja_ouvert
        st.rerun()

    if st.session_state.afficher_details.get(cle, False):
        afficher_details_prediction(predictions, resultats_qualite)

    st.markdown("**🔥 Carte Grad-CAM :**")
    if cle not in st.session_state.gradcam_cache:
        if st.button("🔥 Générer la carte Grad-CAM", key=f"btn_gradcam_{cle}"):
            with st.spinner("Génération en cours..."):
                st.session_state.gradcam_cache[cle] = generate_gradcam(model, img_array)
            st.rerun()
        else:
            st.caption("Calculée à la demande pour ne pas ralentir l'analyse du lot.")
    if cle in st.session_state.gradcam_cache:
        img_orig, heatmap, superimposed = st.session_state.gradcam_cache[cle]
        if superimposed is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(img_orig, caption="Originale", width="stretch")
            with c2:
                st.image(heatmap, caption="Heatmap", width="stretch")
            with c3:
                st.image(superimposed, caption="Superposition", width="stretch")

    st.markdown("**💡 Recommandation clinique :**")
    if statut == 'MALIN':
        st.markdown("- 🏥 Consultez un dermatologue rapidement\n"
                     "- 📋 Apportez ce rapport lors de votre consultation")
    elif statut == 'PRÉCANCÉREUX':
        st.markdown("- 🟡 Lésion précancéreuse — surveillance recommandée")
    else:
        st.markdown("- 🟢 Lésion probablement bénigne — surveillance régulière conseillée")

    if cle in st.session_state.pdf_cache:
        st.download_button(
            f"⬇️ Rapport PDF — {nom_fichier}",
            data=st.session_state.pdf_cache[cle],
            file_name=f"DermIA_{resultat['patient_nom'] or 'patient'}_{nom_fichier.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_pdf_{cle}"
        )
    else:
        if st.button(f"📄 Générer le rapport PDF — {nom_fichier}", key=f"btn_pdf_{cle}", use_container_width=True):
            with st.spinner("Génération du rapport PDF..."):
                if cle not in st.session_state.gradcam_cache:
                    st.session_state.gradcam_cache[cle] = generate_gradcam(model, img_array)
                img_orig, heatmap, superimposed = st.session_state.gradcam_cache[cle]
                st.session_state.pdf_cache[cle] = generer_pdf(
                    resultat["patient_nom"], resultat["patient_age"], nom_classe, code_classe,
                    statut, confidence, predictions, img_orig, heatmap, superimposed
                )
            st.rerun()

# ── Interface principale ───────────────────────────────────
def main():

    st.markdown("""
    <div class="dermia-header">
        <h1>🔬 DermIA</h1>
        <p>Détection intelligente des tumeurs de la peau — EfficientNetB0 · Transfer Learning · Grad-CAM</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ DermIA est un outil d'aide à la décision uniquement. "
        "Il ne remplace pas le diagnostic d'un dermatologue certifié."
    )

    patient_nom, patient_age = sidebar_patient()
    sidebar_statistiques()

    with st.spinner("Chargement du modèle DermIA..."):
        model = charger_modele()
        modele_validation = charger_modele_validation()

    if model is None:
        st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
        st.stop()

    if modele_validation is None:
        st.info(
            "ℹ️ Filtre peau/pas-peau non trouvé "
            f"({VALIDATION_MODEL_PATH.name}) — les images seront directement "
            "envoyées au modèle de diagnostic sans vérification préalable."
        )

    tab_analyse, tab_perf, tab_historique = st.tabs(
        ["🔍 Analyse de lésions", "📊 Performance du modèle", "📋 Historique patients"]
    )

    # ============================================================
    # TAB 1 — ANALYSE (avec upload multi-images / traitement par lot)
    # ============================================================
    with tab_analyse:
        st.success("✓ Modèle DermIA (EfficientNetB0) chargé avec succès")

        afficher_diagramme_7_classes()
        st.markdown("---")
        afficher_regle_abcde()
        st.markdown("---")
        afficher_explication_confiance()
        st.markdown("---")

        st.subheader("📁 Chargez une ou plusieurs images de lésion cutanée")
        st.caption(
            "Sélection multiple possible — toutes les images sont envoyées "
            "au modèle **en une seule fois** (prédiction par lot / batch)."
        )
        images_uploadees = st.file_uploader(
            "Formats acceptés : JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if images_uploadees:
            st.info(f"📦 {len(images_uploadees)} image(s) chargée(s).")
            noms_actuels = tuple(f.name for f in images_uploadees)

            if st.button("🔍 Analyser toutes les images", type="primary", use_container_width=True):

                with st.spinner(f"Prétraitement de {len(images_uploadees)} image(s)..."):
                    images_pil = [Image.open(f) for f in images_uploadees]
                    img_arrays = [preparer_image(img) for img in images_pil]
                    batch_array = np.vstack(img_arrays)

                with st.spinner(f"Prédiction par lot (batch) sur {len(images_uploadees)} image(s)..."):
                    predictions_batch = model.predict(batch_array, verbose=0)
                    scores_peau_batch = None
                    if modele_validation is not None:
                        scores_peau_batch = modele_validation.predict(batch_array, verbose=0).reshape(-1)

                barre = st.progress(0.0, text="Contrôle qualité des images...")
                resultats_lot = []
                for idx, image_uploadee in enumerate(images_uploadees):
                    predictions = predictions_batch[idx]
                    pred_index  = int(np.argmax(predictions))
                    confidence  = float(predictions[pred_index]) * 100
                    nom_classe, code_classe, statut, emoji, color = CLASSES[pred_index]

                    resultats_qualite, qualite_ok = valider_qualite_image(images_pil[idx])

                    if scores_peau_batch is not None:
                        score_peau = float(scores_peau_batch[idx])
                        est_peau = score_peau >= SEUIL_PEAU
                    else:
                        score_peau, est_peau = None, True  # filtre indisponible -> on laisse passer

                    resultats_lot.append({
                        "nom_fichier": image_uploadee.name,
                        "image_pil": images_pil[idx],
                        "img_array": img_arrays[idx],
                        "predictions": predictions,
                        "pred_index": pred_index,
                        "confidence": confidence,
                        "nom_classe": nom_classe,
                        "code_classe": code_classe,
                        "statut": statut,
                        "emoji": emoji,
                        "resultats_qualite": resultats_qualite,
                        "qualite_ok": qualite_ok,
                        "patient_nom": patient_nom,
                        "patient_age": patient_age,
                        "score_peau": score_peau,
                        "est_peau": est_peau,
                    })

                    # Les images non reconnues comme des lésions cutanées ne sont
                    # pas ajoutées à l'historique patient (faux diagnostic évité).
                    if est_peau:
                        st.session_state.historique.append({
                            "nom": patient_nom or "Non renseigné",
                            "age": patient_age,
                            "classe": nom_classe,
                            "code": code_classe,
                            "statut": statut,
                            "confiance": round(confidence, 1),
                            "qualite_image": "OK" if qualite_ok else "Limite",
                            "date": datetime.now().strftime("%d/%m/%Y %H:%M")
                        })
                    barre.progress((idx + 1) / len(images_uploadees),
                                    text=f"Contrôle qualité — {idx + 1}/{len(images_uploadees)}")
                barre.empty()

                sauvegarder_historique(st.session_state.historique)
                # Réinitialise les caches Grad-CAM/PDF (nouveau lot)
                st.session_state.gradcam_cache = {}
                st.session_state.pdf_cache = {}
                st.session_state.afficher_details = {}
                st.session_state.batch_resultats = {
                    "noms": noms_actuels,
                    "resultats": resultats_lot,
                }
                st.success(f"✓ {len(images_uploadees)} image(s) analysée(s) et enregistrée(s) dans l'historique.")

            # Affiche les résultats en cache (persistent même si on clique sur "Détails")
            batch = st.session_state.batch_resultats
            if batch and batch["noms"] == noms_actuels:
                resultats = batch["resultats"]
                st.markdown("---")
                st.subheader(f"📊 Résultats — {len(resultats)} image(s)")

                # Tableau récapitulatif — indispensable pour visualiser un grand lot (ex. 50 images)
                nb_rejetees = sum(1 for r in resultats if not r["est_peau"])
                if nb_rejetees:
                    st.warning(
                        f"⚠️ {nb_rejetees} image(s) ne ressemble(nt) pas à une lésion cutanée "
                        "et n'ont pas été ajoutées à l'historique patient (voir colonne "
                        "« Type d'image » ci-dessous)."
                    )

                lignes_recap = [{
                    "Fichier": r["nom_fichier"],
                    "Type d'image": "✓ Peau" if r["est_peau"] else "⚠️ Pas une lésion cutanée",
                    "Classe prédite": f"{r['emoji']} {r['code_classe']} — {r['nom_classe']}" if r["est_peau"] else "—",
                    "Statut": r["statut"] if r["est_peau"] else "—",
                    "Confiance": f"{r['confidence']:.1f}%" if r["est_peau"] else "—",
                    "Qualité image": "✓ OK" if r["qualite_ok"] else "⚠️ Limite",
                } for r in resultats]
                st.dataframe(pd.DataFrame(lignes_recap), use_container_width=True, hide_index=True)
                st.caption(
                    "Déplie chaque image ci-dessous pour voir les détails, "
                    "le Grad-CAM et générer le rapport PDF correspondant."
                )

                expanded_defaut = len(resultats) <= 3
                for idx, resultat in enumerate(resultats):
                    if resultat["est_peau"]:
                        titre = (f"{resultat['emoji']} {idx + 1}/{len(resultats)} — {resultat['nom_fichier']} — "
                                 f"{resultat['nom_classe']} ({resultat['confidence']:.1f}%)")
                    else:
                        titre = f"⚠️ {idx + 1}/{len(resultats)} — {resultat['nom_fichier']} — Pas une lésion cutanée"
                    with st.expander(titre, expanded=expanded_defaut):
                        afficher_resultat_image(idx, resultat, model)

    # ============================================================
    # TAB 2 — PERFORMANCE DU MODÈLE
    # ============================================================
    with tab_perf:
        afficher_performance_modele()

    # ============================================================
    # TAB 3 — HISTORIQUE PATIENTS
    # ============================================================
    with tab_historique:
        if st.session_state.historique:
            st.subheader("📋 Historique complet des patients")
            df_hist = pd.DataFrame(st.session_state.historique)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                csv_export = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Exporter l'historique (CSV)",
                    data=csv_export,
                    file_name=f"DermIA_historique_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                if st.button("🗑️ Vider l'historique", use_container_width=True):
                    st.session_state.historique = []
                    st.session_state.batch_resultats = None
                    st.session_state.afficher_details = {}
                    st.session_state.gradcam_cache = {}
                    st.session_state.pdf_cache = {}
                    sauvegarder_historique([])
                    st.rerun()
        else:
            st.info("Aucune analyse enregistrée pour le moment.")

    st.markdown("---")
    st.markdown(
        "**DermIA** — Projet de fin d'études : système intelligent pour la "
        "détection précoce des tumeurs de la peau  \n"
        "Modèle : **EfficientNetB0 + Transfer Learning**  \n"
        "Dataset : **HAM10000** (Tschandl et al., 2018)  \n"
        "Interprétabilité : **Grad-CAM** (Selvaraju et al., 2017)  \n"
        "Auteur : Ramatoulaye Sy — 2026"
    )

if __name__ == "__main__":
    main()
