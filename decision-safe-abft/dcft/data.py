"""Image loading with the DeiT-Tiny evaluation preprocessing."""
import os
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


def list_images(root):
    """All images under root in a fixed, seeded order. Label = parent folder
    name if it is an integer (ImageNetV2 layout), else -1."""
    items = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(EXTS):
                p = os.path.join(d, f)
                lab = os.path.basename(d)
                items.append((p, int(lab) if lab.isdigit() else -1))
    items.sort(key=lambda t: t[0])
    # fixed shuffle so that calibration and test splits cover many classes
    order = np.random.default_rng(0).permutation(len(items))
    return [items[i] for i in order]


def load_image(path):
    """Resize shorter side to 256 (bicubic), centre-crop 224, normalise.
    Returns float32 (3, 224, 224)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256.0 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)
