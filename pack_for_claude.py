"""Pack the DeiT-Tiny weights and a small set of test images into one zip,
so the new certificate can be developed against the real model.

Run from the decision-safe-abft folder (the one with dcft/ and weights/):
    python pack_for_claude.py --images "<your ImageNetV2 folder>"
It writes for_claude.zip (about 30 MB): the weights, 80 images, and the
paper's E2/E3 results if they are in results/.
"""
import argparse, os, zipfile
from dcft.data import list_images

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--n", type=int, default=80)
a = ap.parse_args()
items = list_images(a.images)
cal = items[:40]                 # first images = calibration split
test = items[200:200 + a.n - 40]  # start of the test split
with zipfile.ZipFile("for_claude.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("weights/deit_tiny.npz", "weights/deit_tiny.npz")
    for extra in ("results/e2.npz", "results/calib.json", "results/e3_injections.csv"):
        if os.path.exists(extra):
            z.write(extra, "old_" + extra.replace("/", "_"))
    for split, its in (("cal", cal), ("test", test)):
        for k, (p, lab) in enumerate(its):
            z.write(p, f"images/{split}/{lab}/{k:03d}_{os.path.basename(p)}")
print("wrote for_claude.zip with", len(cal) + len(test), "images")
