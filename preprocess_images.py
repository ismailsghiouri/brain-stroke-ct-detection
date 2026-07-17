"""
Prétraitement du dataset Brain_Data_Organised (Normal vs Stroke) avant
entraînement d'un CNN.

Étapes :
1. Redimensionnement de toutes les images à 224x224 (au lieu de 650x650)
2. Normalisation des pixels entre 0 et 1
3. Split stratifié train/validation/test (70/15/15)
4. Data augmentation sur le train uniquement, plus marquée sur "Stroke"
   (classe minoritaire) pour réduire le déséquilibre 1551/950 (~1.63)

Sorties (dans preprocessed_data/) :
  train.npz  -> X_train (N,224,224,3) float32 dans [0,1], y_train (N,)
  val.npz    -> X_val,  y_val
  test.npz   -> X_test, y_test

0 = Normal, 1 = Stroke
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration -----------------------------------------------------
DATA_DIR = "Brain_Data_Organised"
CLASSES = ["Normal", "Stroke"]  # index = label (0, 1)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

IMG_SIZE = (224, 224)  # taille cible pour le CNN
OUTPUT_DIR = "preprocessed_data"

# Proportions du split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Nombre de copies augmentées ajoutées PAR IMAGE ORIGINALE, uniquement pour
# le train. Stroke (label 1) reçoit plus de copies que Normal (label 0)
# pour resserrer l'écart entre les deux classes après augmentation.
AUG_COPIES_PER_IMAGE = {0: 1, 1: 2}  # Normal: +1 copie, Stroke: +2 copies

RANDOM_SEED = 42


# --- Étape 0 : lister les images ---------------------------------------
def gather_filepaths_labels():
    """Retourne (paths, labels) pour toutes les images du dataset."""
    paths, labels = [], []
    for label, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return paths, np.array(labels)


# --- Étapes 1 & 2 : resize + normalisation -------------------------------
def load_image_resized(path, size=IMG_SIZE):
    """Charge une image et la redimensionne (retourne un objet PIL.Image)."""
    with Image.open(path) as img:
        # convert("RGB") garantit 3 canaux, même si le jpg est en niveaux de gris
        return img.convert("RGB").resize(size, Image.LANCZOS)


def to_normalized_array(img):
    """Convertit une image PIL en tableau numpy float32 normalisé [0, 1]."""
    return np.asarray(img, dtype=np.float32) / 255.0


# --- Étape 3 : split stratifié -------------------------------------------
def stratified_split(paths, labels):
    """Split 70/15/15 en conservant la proportion Normal/Stroke dans chaque
    ensemble (stratify=labels)."""
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    # Le reste (30%) est divisé moitié/moitié -> 15% val, 15% test
    relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )
    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


# --- Étape 4 : data augmentation (train uniquement) ----------------------
def make_augmented_variant(img, rng):
    """Applique une combinaison aléatoire de flip horizontal, rotation légère
    et léger zoom à une image PIL. Retourne une nouvelle image PIL."""
    out = img

    # Flip horizontal (50% de chance)
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotation légère : +/- 15 degrés
    angle = rng.uniform(-15, 15)
    out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # Léger zoom avant (jusqu'à 15%) : on recadre le centre puis on
    # redimensionne à la taille d'origine
    zoom_factor = rng.uniform(1.0, 1.15)
    if zoom_factor > 1.0:
        w, h = out.size
        new_w, new_h = w / zoom_factor, h / zoom_factor
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        out = out.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)

    return out


def build_split_arrays(paths, labels, augment=False, rng=None):
    """Construit les tableaux X (images normalisées) et y (labels) pour un
    split donné. Si augment=True, ajoute des copies augmentées par classe
    selon AUG_COPIES_PER_IMAGE (utilisé seulement pour le train)."""
    X_list, y_list = [], []

    for path, label in zip(paths, labels):
        base_img = load_image_resized(path)
        X_list.append(to_normalized_array(base_img))
        y_list.append(label)

        if augment:
            n_copies = AUG_COPIES_PER_IMAGE.get(label, 0)
            for _ in range(n_copies):
                aug_img = make_augmented_variant(base_img, rng)
                X_list.append(to_normalized_array(aug_img))
                y_list.append(label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def print_class_counts(title, y):
    counts = {label: int(np.sum(y == label)) for label in range(len(CLASSES))}
    total = len(y)
    parts = [f"{CLASSES[label]}={count}" for label, count in counts.items()]
    ratio = max(counts.values()) / max(min(counts.values()), 1)
    print(f"  {title}: total={total} ({', '.join(parts)}) | ratio={ratio:.2f}")


def save_augmentation_preview(paths, labels, rng):
    """Sauvegarde un aperçu visuel : image originale vs. versions augmentées,
    pour une image Normal et une image Stroke."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, label in enumerate([0, 1]):
        idx = int(np.where(labels == label)[0][0])
        base_img = load_image_resized(paths[idx])

        axes[row, 0].imshow(base_img)
        axes[row, 0].set_title(f"{CLASSES[label]} - original")
        for col in range(1, 3):
            aug_img = make_augmented_variant(base_img, rng)
            axes[row, col].imshow(aug_img)
            axes[row, col].set_title(f"{CLASSES[label]} - augmentée {col}")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=120)
    plt.close(fig)
    print("  Aperçu de l'augmentation sauvegardé dans 'augmentation_preview.png'")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=== Chargement de la liste des images ===")
    paths, labels = gather_filepaths_labels()
    print(f"  {len(paths)} images trouvées ({CLASSES[0]}={int(np.sum(labels == 0))}, "
          f"{CLASSES[1]}={int(np.sum(labels == 1))})")

    print("\n=== Split stratifié train/val/test (70/15/15) ===")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_split(
        paths, labels
    )
    print_class_counts("Train (avant augmentation)", train_labels)
    print_class_counts("Val", val_labels)
    print_class_counts("Test", test_labels)

    print("\n=== Aperçu de la data augmentation ===")
    save_augmentation_preview(train_paths, train_labels, rng)

    print("\n=== Construction des tableaux (resize 224x224 + normalisation) ===")
    print("  Train (avec augmentation)...")
    X_train, y_train = build_split_arrays(train_paths, train_labels, augment=True, rng=rng)
    print("  Validation (sans augmentation)...")
    X_val, y_val = build_split_arrays(val_paths, val_labels, augment=False)
    print("  Test (sans augmentation)...")
    X_test, y_test = build_split_arrays(test_paths, test_labels, augment=False)

    print("\n=== Résumé final ===")
    print_class_counts("Train (après augmentation)", y_train)
    print_class_counts("Val", y_val)
    print_class_counts("Test", y_test)
    print(f"  Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  Valeurs de pixels dans [{X_train.min():.2f}, {X_train.max():.2f}]")

    print("\n=== Sauvegarde des fichiers .npz ===")
    np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"), X=X_test, y=y_test)
    print(f"  Fichiers écrits dans '{OUTPUT_DIR}/' (train.npz, val.npz, test.npz)")


if __name__ == "__main__":
    main()
