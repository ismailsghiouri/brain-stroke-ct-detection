"""
Premier CNN (baseline) pour classifier des CT cérébraux : Normal vs Stroke.

Entraîné sur les données déjà prétraitées par preprocess_images.py
(preprocessed_data/{train,val,test}.npz -> images 224x224 normalisées [0,1]).

0 = Normal, 1 = Stroke
"""

import os

# Ce projet tourne dans un venv Python 3.12 dédié (.venv) avec le vrai TensorFlow
# (Python 3.14, la version par défaut de la machine, n'a pas encore de wheel TF).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
# Sur certains CPU (ex: Tiger Lake), le chemin d'optimisation oneDNN de TF peut se
# bloquer indéfiniment à l'import. On le désactive ; impact négligeable sur un
# aussi petit modèle. Ces deux lignes doivent s'exécuter AVANT `import tensorflow`.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "preprocessed_data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_baseline.keras")

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 30

# Au-delà de ce ratio (classe majoritaire / minoritaire) dans le train, on
# active class_weight en plus de l'augmentation.
CLASS_WEIGHT_RATIO_THRESHOLD = 1.2


def load_split(name):
    data = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return data["X"], data["y"]


def build_model():
    """3-4 blocs Conv2D+MaxPooling2D (filtres croissants) -> GlobalAveragePooling2D
    -> Dense -> sortie sigmoid. Voir l'explication détaillée envoyée au chat."""
    model = keras.Sequential(
        [
            layers.Input(shape=IMG_SHAPE),

            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_baseline",
    )
    return model


def get_class_weight(y_train):
    """Décide si class_weight est encore utile vu le déséquilibre résiduel
    après data augmentation (~1.09 attendu)."""
    counts = np.bincount(y_train)
    ratio = counts.max() / counts.min()
    print(f"  Ratio résiduel dans le train (après augmentation) : {ratio:.2f}")

    if ratio <= CLASS_WEIGHT_RATIO_THRESHOLD:
        print(
            f"  Ratio <= {CLASS_WEIGHT_RATIO_THRESHOLD} : la data augmentation a déjà "
            "quasiment équilibré les classes. Ajouter class_weight ici ne ferait que "
            "pondérer très légèrement (~±5%) une différence déjà négligeable, sans "
            "bénéfice réel. On ne l'utilise donc pas."
        )
        return None

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {label: weight for label, weight in zip(np.unique(y_train), weights)}
    print(f"  Ratio > {CLASS_WEIGHT_RATIO_THRESHOLD} : class_weight activé -> {class_weight}")
    return class_weight


def plot_history(history):
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs_range, acc, marker="o", label="Train")
    axes[0].plot(epochs_range, val_acc, marker="o", label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, marker="o", label="Train")
    axes[1].plot(epochs_range, val_loss, marker="o", label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("  Courbes sauvegardées dans 'training_curves.png'")
    plt.show()


def main():
    print(f"=== Backend Keras utilisé : {keras.backend.backend()} ===")

    print("\n=== Chargement des données prétraitées ===")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n=== Déséquilibre résiduel des classes ===")
    class_weight = get_class_weight(y_train)

    print("\n=== Construction du modèle ===")
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("\n=== Entraînement ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )

    print("\n=== Évaluation sur le jeu de test ===")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss:      {test_loss:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test precision: {test_prec:.4f}")
    print(f"  Test recall:    {test_rec:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n  Modèle sauvegardé dans '{MODEL_PATH}'")

    print("\n=== Courbes d'entraînement ===")
    plot_history(history)


if __name__ == "__main__":
    main()
