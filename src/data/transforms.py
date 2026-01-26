from __future__ import annotations

from torchvision import transforms


def build_train_transforms(image_size: int) -> transforms.Compose:
    # Minimal for Week 1 baseline. Augmentation comes Week 2.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_val_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
