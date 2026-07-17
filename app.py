"""
Application Streamlit de démonstration : détection d'AVC sur CT cérébral.

Charge le CNN entraîné (models/cnn_baseline.keras), applique exactement le
même prétraitement qu'à l'entraînement, prédit Normal/Stroke, et explique la
décision avec une carte de chaleur Grad-CAM.

Outil éducatif uniquement -- voir l'avertissement affiché dans l'app.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/cnn_baseline.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Stroke"]
EXAMPLE_IMAGES = {
    "Normal": "examples/exemple_normal.jpg",
    "Stroke": "examples/exemple_stroke.jpg",
}


# --------------------------------------------------------------------------
# Chargement du modèle
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Charge le modèle une seule fois et le garde en mémoire entre les
    interactions (sans ce cache, Streamlit relancerait tout le script -- donc
    rechargerait le modèle -- à chaque clic)."""
    return keras.models.load_model(MODEL_PATH)


# --------------------------------------------------------------------------
# Prétraitement
# --------------------------------------------------------------------------
def preprocess_image(pil_image):
    """Reproduit exactement le prétraitement de preprocess_images.py :
    conversion en RGB, resize à 224x224, pixels ramenés entre 0 et 1.
    Un modèle est très sensible à ça : s'entraîner sur des pixels [0, 1] puis
    lui donner des pixels [0, 255] au moment de prédire fausserait tout.
    """
    resized = pil_image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)  # ajoute la dimension de batch : (1, 224, 224, 3)


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
def predict(model, preprocessed_image):
    """Le modèle a une seule sortie sigmoid : un score entre 0 et 1 qui
    représente la probabilité de la classe "Stroke". On le compare au seuil
    0.5 pour décider de la classe, puis on convertit ce score en un
    pourcentage de confiance dans la classe prédite (pas juste dans "Stroke")."""
    raw_score = float(model.predict(preprocessed_image, verbose=0)[0][0])
    predicted_index = int(raw_score >= 0.5)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = raw_score if predicted_index == 1 else 1 - raw_score
    return predicted_label, confidence * 100


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
def find_last_conv_layer(model):
    """Repère automatiquement la dernière couche Conv2D du modèle. C'est la
    dernière étape où l'information spatiale ("quelle zone de l'image")
    existe encore -- après elle, GlobalAveragePooling2D résume tout en un
    seul vecteur et on perd cette notion de position."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("Aucune couche Conv2D trouvée dans ce modèle.")


def make_gradcam_heatmap(preprocessed_image, model, last_conv_layer_name):
    """Génère une carte de chaleur Grad-CAM (Selvaraju et al., 2017).

    Principe simplifié : on calcule le gradient du score "Stroke" par rapport
    à la dernière carte de features convolutive. Ce gradient indique, pour
    chaque zone de l'image, "si l'activation ici augmentait, le score Stroke
    augmenterait-il aussi, et de combien ?". Les zones à fort gradient positif
    sont celles qui ont le plus poussé le modèle vers sa décision -- ce sont
    elles qu'on colore en rouge/jaune sur la heatmap finale.

    Note technique : on rejoue les couches une par une "à la main" dans la
    boucle de gradient, plutôt que de découper le modèle en sous-modèle. Avec
    un modèle Sequential rechargé depuis un fichier .keras, cette dernière
    approche (pourtant standard pour un modèle Functional) fait perdre la
    connexion du gradient (tape.gradient renvoie None).
    """
    x = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)
    conv_output = None
    with tf.GradientTape() as tape:
        tape.watch(x)
        h = x
        for layer in model.layers:
            h = layer(h, training=False)
            if layer.name == last_conv_layer_name:
                conv_output = h
        score = h[:, 0]  # sortie sigmoid unique : le score Stroke lui-même

    grads = tape.gradient(score, conv_output)
    # Importance moyenne de chaque filtre de la dernière couche conv --
    # c'est le coeur de l'algorithme Grad-CAM.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # On ne garde que les contributions positives (ce qui pousse vers
    # "Stroke"), puis on ramène tout entre 0 et 1 pour l'afficher comme une image.
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    """Superpose la heatmap (colorée avec la palette 'jet' : bleu = faible
    influence, rouge = forte influence) sur l'image originale."""
    original = pil_image.convert("RGB")

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_uint8).resize(original.size, Image.BILINEAR)
    heatmap_normalized = np.array(heatmap_resized) / 255.0

    jet = matplotlib.colormaps["jet"]
    colored_heatmap = np.uint8(jet(heatmap_normalized)[:, :, :3] * 255)
    colored_heatmap_img = Image.fromarray(colored_heatmap)

    return Image.blend(original, colored_heatmap_img, alpha)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
st.set_page_config(page_title="Détection d'AVC - CT cérébral", page_icon="🧠")

st.error(
    "⚠️ Outil éducatif réalisé dans le cadre d'un projet étudiant. "
    "Ne remplace en aucun cas un diagnostic médical. "
    "Ne doit pas être utilisé pour de vraies décisions cliniques."
)

st.title("🧠 Détection d'AVC sur imagerie cérébrale (CT)")
st.write(
    "Cette application utilise un CNN entraîné pour classifier des scanners "
    "cérébraux (CT) en deux catégories : **Normal** ou **Stroke** (AVC)."
)

st.subheader("Pas d'image sous la main ?")
st.write("Télécharge un exemple de chaque classe pour tester l'application :")
with st.container(horizontal=True):
    for class_name, path in EXAMPLE_IMAGES.items():
        with open(path, "rb") as example_file:
            st.download_button(
                f"Exemple {class_name}",
                data=example_file.read(),
                file_name=os.path.basename(path),
                mime="image/jpeg",
                icon=":material/download:",
            )

st.subheader("Analyser une image")
uploaded_file = st.file_uploader(
    "Scanner cérébral (CT) au format JPG ou PNG",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
    except Exception:
        st.error("Impossible de lire ce fichier comme une image. Essaie un autre fichier JPG/PNG.")
        st.stop()

    model = load_model()
    preprocessed = preprocess_image(pil_image)
    label, confidence = predict(model, preprocessed)

    with st.spinner("Génération de la carte Grad-CAM..."):
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(preprocessed, model, last_conv_layer_name)
        overlaid_image = overlay_heatmap(pil_image, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Image originale", width="stretch")
    with col2:
        st.image(overlaid_image, caption="Zones ayant influencé la décision (Grad-CAM)", width="stretch")

    st.subheader("Résultat")
    if label == "Stroke":
        st.error(f"🔴 **Stroke détecté** -- confiance : {confidence:.1f}%")
        st.error("**Risque : élevé** -- une anomalie compatible avec un AVC a été détectée.")
    else:
        st.success(f"🟢 **Normal** -- confiance : {confidence:.1f}%")
        st.success("**Risque : faible** -- aucune anomalie détectée par le modèle.")

    st.caption(
        "Rappel : cette prédiction provient d'un modèle expérimental et ne "
        "constitue en aucun cas un avis médical."
    )

with st.expander("ℹ️ À propos de ce projet"):
    st.markdown(
        """
**Dataset** : Brain CT images dataset (Kaggle), organisé en deux classes,
`Normal` et `Stroke`, environ 2500 images de CT cérébraux au total.

**Modèle déployé** : un CNN simple entraîné from scratch (4 blocs Conv2D +
MaxPooling2D, filtres croissants 32 → 64 → 128 → 256, puis
GlobalAveragePooling2D et une couche dense avant la sortie).

**Performance sur le jeu de test (376 images)** :
- Accuracy : 93.88%
- Precision : 88.96%
- Recall : 95.80%

Le recall est la métrique la plus surveillée ici : elle mesure la capacité du
modèle à ne pas rater un vrai cas de Stroke (un faux négatif -- un AVC non
détecté -- est le pire scénario possible dans un contexte médical).

**Résultat intéressant** : une tentative de transfer learning avec
MobileNetV2 (pré-entraîné sur ImageNet) a aussi été testée, mais a obtenu de
moins bons résultats sur toutes les métriques (accuracy 91.22%, recall
90.91%) que ce CNN pourtant entraîné from scratch. Explication plausible :
les filtres appris sur des photos naturelles (ImageNet) ne sont pas
forcément les plus adaptés à des images médicales en niveaux de gris, et ce
dataset s'est révélé suffisant pour qu'un petit CNN dédié apprenne
directement de bons filtres.

**Limites à garder en tête** :
- Dataset de taille restreinte (~2500 images), provenant d'une seule source
- Aucune validation clinique : ce modèle n'a jamais été évalué par des
  radiologues ni testé sur des données d'hôpital réelles
- Les images sont des CT scans, pas des IRM -- un modèle entraîné sur l'une
  de ces modalités ne se généralise pas forcément à l'autre
- Le Grad-CAM montre où le modèle a "regardé", pas nécessairement pourquoi
  cette zone est cliniquement pertinente
        """
    )
