"""
Transfer learning avec MobileNetV2 (pré-entraîné sur ImageNet) pour classifier
des CT cérébraux : Normal vs Stroke. À comparer avec le CNN baseline
(train_cnn.py -> models/cnn_baseline.keras).

Entraînement en 2 phases (voir l'explication envoyée au chat) :
  Phase 1 : base MobileNetV2 gelée, on entraîne uniquement la nouvelle tête
  Phase 2 : on dégèle les dernières couches de MobileNetV2 et on affine
            (fine-tuning) avec un learning rate beaucoup plus faible

Données réutilisées telles quelles depuis preprocessed_data/{train,val,test}.npz
(images 224x224 normalisées en [0, 1]).

0 = Normal, 1 = Stroke
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
from keras import layers
from keras.applications import MobileNetV2

DATA_DIR = "preprocessed_data"
MODEL_DIR = "models"
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_baseline.keras")
TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, "transfer_learning.keras")

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

PHASE1_EPOCHS = 10  # tête seule : converge généralement vite
PHASE1_LR = 1e-3

PHASE2_EPOCHS = 20  # fine-tuning : LR faible donc progression plus lente
PHASE2_LR = 1e-5  # ~100x plus faible que la phase 1

# On ne dégèle que les dernières couches de MobileNetV2 (les plus "spécialisées").
# Les couches du début (features génériques : bords, textures) restent gelées.
FINE_TUNE_LAYER_RATIO = 0.8  # gèle les 80% premières couches, dégèle les 20% dernières


def load_split(name):
    data = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return data["X"], data["y"]


def build_model():
    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model.trainable = False  # phase 1 : base entièrement gelée

    inputs = keras.Input(shape=IMG_SHAPE)
    # Nos images sont déjà normalisées en [0, 1] (preprocess_images.py). MobileNetV2
    # attend des pixels en [-1, 1] (son préprocessing ImageNet standard). Cette couche
    # fait exactement la même transformation, adaptée à notre entrée déjà /255.
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    # training=False est volontairement figé ici (indépendant de base_model.trainable) :
    # ça force les BatchNorm de MobileNetV2 à toujours utiliser leurs statistiques
    # ImageNet (moving mean/variance), même pendant le fine-tuning en phase 2. Seuls
    # les poids (gamma/beta et les filtres convolutifs) des couches dégelées seront mis
    # à jour ; les statistiques de normalisation, elles, ne bougent jamais.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def make_early_stopping():
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


def print_comparison_table(baseline_metrics, transfer_metrics):
    headers = ["Metrique", "CNN baseline", "Transfer learning", "Delta"]
    rows = []
    for key, label in [
        ("loss", "Test loss"),
        ("accuracy", "Test accuracy"),
        ("precision", "Test precision"),
        ("recall", "Test recall"),
    ]:
        b, t = baseline_metrics[key], transfer_metrics[key]
        rows.append((label, f"{b:.4f}", f"{t:.4f}", f"{t - b:+.4f}"))

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols):
        return " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))


def main():
    print("=== Chargement des données prétraitées ===")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n=== Construction du modèle (MobileNetV2 gelé + tête) ===")
    model, base_model = build_model()
    compile_model(model, PHASE1_LR)
    model.summary()

    print(f"\n=== Phase 1/2 : entraînement de la tête, base gelée (jusqu'à {PHASE1_EPOCHS} epochs) ===")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[make_early_stopping()],
    )

    print("\n=== Dégel des dernières couches de MobileNetV2 pour le fine-tuning ===")
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * FINE_TUNE_LAYER_RATIO)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    n_trainable_layers = sum(layer.trainable for layer in base_model.layers)
    print(
        f"  {n_trainable_layers}/{len(base_model.layers)} couches de MobileNetV2 dégelées "
        f"(à partir de la couche {fine_tune_at})"
    )

    compile_model(model, PHASE2_LR)  # recompiler est obligatoire après un changement de `trainable`
    model.summary()

    print(f"\n=== Phase 2/2 : fine-tuning, lr={PHASE2_LR} (jusqu'à {PHASE2_EPOCHS} epochs) ===")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PHASE2_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[make_early_stopping()],
    )

    print("\n=== Évaluation sur le jeu de test (transfer learning) ===")
    tl_loss, tl_acc, tl_prec, tl_rec = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss:      {tl_loss:.4f}")
    print(f"  Test accuracy:  {tl_acc:.4f}")
    print(f"  Test precision: {tl_prec:.4f}")
    print(f"  Test recall:    {tl_rec:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(TRANSFER_MODEL_PATH)
    print(f"\n  Modèle sauvegardé dans '{TRANSFER_MODEL_PATH}'")

    print("\n=== Comparaison avec le CNN baseline ===")
    if os.path.exists(BASELINE_MODEL_PATH):
        baseline_model = keras.models.load_model(BASELINE_MODEL_PATH)
        bl_loss, bl_acc, bl_prec, bl_rec = baseline_model.evaluate(X_test, y_test, verbose=0)
        print_comparison_table(
            {"loss": bl_loss, "accuracy": bl_acc, "precision": bl_prec, "recall": bl_rec},
            {"loss": tl_loss, "accuracy": tl_acc, "precision": tl_prec, "recall": tl_rec},
        )
    else:
        print(f"  '{BASELINE_MODEL_PATH}' introuvable : impossible de générer la comparaison.")


if __name__ == "__main__":
    main()
