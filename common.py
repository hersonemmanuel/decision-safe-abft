"""Shared helpers for the experiment scripts."""
import numpy as np

from dcft.data import list_images, load_image


def image_source(images_dir, synthetic, offset, count):
    """Yield (image, label) for `count` images starting at `offset`.

    images_dir: folder of images (ImageNetV2 layout recommended), used in
    sorted order so that calibration and test splits never overlap.
    synthetic: if > 0, ignore images_dir and yield seeded random images
    (only for the self-test).
    """
    if synthetic:
        for k in range(offset, offset + count):
            r = np.random.default_rng(1000 + k)
            yield r.normal(0, 1, (3, 224, 224)).astype(np.float32), -1
        return
    items = list_images(images_dir)
    if offset + count > len(items):
        raise SystemExit(f"need {offset + count} images, found {len(items)} in {images_dir}")
    for p, lab in items[offset:offset + count]:
        yield load_image(p), lab
