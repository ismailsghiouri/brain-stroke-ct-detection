"""
Exploration du dataset Brain_Data_Organised (Normal vs Stroke).

Ce script répond à 4 questions de base avant tout entraînement de modèle :
1. Combien d'images par classe ?
2. À quoi ressemblent-elles ?
3. Ont-elles toutes la même taille ?
4. Les classes sont-elles équilibrées ?
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration -----------------------------------------------------
DATA_DIR = "Brain_Data_Organised"
CLASSES = ["Normal", "Stroke"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
N_EXAMPLES = 4  # nombre d'images d'exemple à afficher par classe


def list_images(class_name):
    """Retourne la liste des chemins d'images valides pour une classe."""
    class_dir = os.path.join(DATA_DIR, class_name)
    return [
        os.path.join(class_dir, fname)
        for fname in sorted(os.listdir(class_dir))
        if fname.lower().endswith(VALID_EXTENSIONS)
    ]


def step1_count_images(images_by_class):
    """Étape 1 : compter le nombre d'images par dossier/classe."""
    print("\n=== Étape 1 : Nombre d'images par classe ===")
    total = 0
    for class_name, paths in images_by_class.items():
        print(f"  {class_name}: {len(paths)} images")
        total += len(paths)
    print(f"  Total: {total} images")


def step2_show_examples(images_by_class, n_examples=N_EXAMPLES):
    """Étape 2 : afficher quelques exemples de chaque classe côte à côte."""
    print("\n=== Étape 2 : Affichage d'exemples (fenêtre matplotlib) ===")
    n_classes = len(images_by_class)
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(3 * n_examples, 3 * n_classes))

    for row, (class_name, paths) in enumerate(images_by_class.items()):
        sample_paths = paths[:n_examples]
        for col in range(n_examples):
            ax = axes[row, col] if n_classes > 1 else axes[col]
            if col < len(sample_paths):
                img = Image.open(sample_paths[col])
                ax.imshow(img, cmap="gray")
                ax.set_title(f"{class_name}\n{img.size[0]}x{img.size[1]}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("examples_preview.png", dpi=120)
    print("  Aperçu sauvegardé dans 'examples_preview.png'")
    plt.show()


def step3_check_dimensions(images_by_class):
    """Étape 3 : vérifier si toutes les images ont la même taille."""
    print("\n=== Étape 3 : Vérification des dimensions ===")
    all_sizes = Counter()

    for class_name, paths in images_by_class.items():
        class_sizes = Counter()
        for path in paths:
            with Image.open(path) as img:
                class_sizes[img.size] += 1  # (width, height)
        all_sizes.update(class_sizes)

        print(f"\n  {class_name}: {len(class_sizes)} taille(s) différente(s)")
        for size, count in class_sizes.most_common(5):
            print(f"    {size[0]}x{size[1]}: {count} images")
        if len(class_sizes) > 5:
            print(f"    ... et {len(class_sizes) - 5} autres tailles")

    if len(all_sizes) == 1:
        print("\n  => Toutes les images ont EXACTEMENT la même taille.")
    else:
        print(f"\n  => Les images ont {len(all_sizes)} tailles différentes au total.")
        print("     Il faudra donc redimensionner (resize) toutes les images")
        print("     à une taille commune avant de les donner à un modèle.")

    return all_sizes


def step4_check_balance(images_by_class):
    """Étape 4 : vérifier s'il y a un déséquilibre entre les classes."""
    print("\n=== Étape 4 : Déséquilibre des classes ===")
    counts = {name: len(paths) for name, paths in images_by_class.items()}
    total = sum(counts.values())

    for name, count in counts.items():
        pct = 100 * count / total
        print(f"  {name}: {count} images ({pct:.1f}%)")

    majority = max(counts.values())
    minority = min(counts.values())
    ratio = majority / minority
    print(f"\n  Ratio classe majoritaire / minoritaire: {ratio:.2f}")

    if ratio > 1.5:
        print("  => Déséquilibre notable détecté.")
        print("     Pistes possibles : class_weight, oversampling/undersampling,")
        print("     data augmentation sur la classe minoritaire.")
    else:
        print("  => Les classes sont relativement équilibrées.")

    # Petit graphique en barres pour visualiser le déséquilibre
    plt.figure(figsize=(5, 4))
    plt.bar(counts.keys(), counts.values(), color=["steelblue", "indianred"])
    plt.title("Nombre d'images par classe")
    plt.ylabel("Nombre d'images")
    for i, (name, count) in enumerate(counts.items()):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("class_balance.png", dpi=120)
    print("  Graphique sauvegardé dans 'class_balance.png'")
    plt.show()


def main():
    images_by_class = {class_name: list_images(class_name) for class_name in CLASSES}

    step1_count_images(images_by_class)
    step2_show_examples(images_by_class)
    step3_check_dimensions(images_by_class)
    step4_check_balance(images_by_class)


if __name__ == "__main__":
    main()
