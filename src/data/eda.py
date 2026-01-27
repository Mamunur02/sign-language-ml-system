from pathlib import Path
from collections import Counter
from PIL import Image
import random

DATA_ROOT = Path("data/raw/asl_alphabet/train")


def count_images_per_class(root: Path) -> Counter:
    counts = Counter()
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = sum(1 for p in class_dir.iterdir() if p.is_file())
        counts[class_dir.name] = n
    return counts


def inspect_random_images(root: Path, n: int = 20, seed: int = 42):
    rng = random.Random(seed)
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    image_paths = []
    for cd in class_dirs:
        image_paths.extend([p for p in cd.iterdir() if p.is_file()])

    rng.shuffle(image_paths)
    sample = image_paths[:n]

    sizes = []
    modes = []
    bad = []

    for p in sample:
        try:
            with Image.open(p) as img:
                sizes.append(img.size)   # (width, height)
                modes.append(img.mode)   # e.g. RGB, L
        except Exception as e:
            bad.append((str(p), str(e)))

    return sizes, modes, bad, sample


def check_corrupt_images(root: Path, max_per_class: int = 200, seed: int = 42):
    rng = random.Random(seed)
    bad = []

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = [p for p in class_dir.iterdir() if p.is_file()]
        rng.shuffle(files)
        files = files[:max_per_class]

        for p in files:
            try:
                with Image.open(p) as img:
                    img.verify()  # quick corruption check
            except Exception as e:
                bad.append((class_dir.name, str(p), str(e)))

    return bad


if __name__ == "__main__":
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT.resolve()}")

    print(f"Dataset root: {DATA_ROOT.resolve()}\n")

    counts = count_images_per_class(DATA_ROOT)
    total = sum(counts.values())

    print("---- Class counts ----")
    for cls, n in counts.most_common():
        print(f"{cls:>8}: {n}")
    print(f"\nTotal images: {total}")
    print(f"Num classes: {len(counts)}\n")

    sizes, modes, bad, sample = inspect_random_images(DATA_ROOT, n=30)
    print("---- Random sample inspection (30 images) ----")
    print("Unique sizes:", sorted(set(sizes))[:10], ("..." if len(set(sizes)) > 10 else ""))
    print("Mode counts:", dict(Counter(modes)))
    if bad:
        print("\nErrors opening some sample images:")
        for p, e in bad:
            print(" ", p, "|", e)
    else:
        print("No errors opening sampled images.\n")

    bad2 = check_corrupt_images(DATA_ROOT, max_per_class=200)
    print("---- Corruption check (up to 200 images per class) ----")
    if bad2:
        print(f"Found {len(bad2)} potentially corrupt images. First 10:")
        for row in bad2[:10]:
            print(row)
    else:
        print("No corrupt images detected in sampled check.")
