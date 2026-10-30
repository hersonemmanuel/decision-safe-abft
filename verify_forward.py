"""Check that the NumPy reference forward matches timm exactly (float64).

Run after export_weights.py. A maximum logit difference of about 1e-9 or
smaller means the NumPy model is the same network; anything above 1e-6
means the mapping of the architecture is wrong and the experiments must not
be run until it is fixed.
"""
import argparse
import numpy as np
import torch
import timm

from dcft.model import Weights, ref_forward, Linear
from common import image_source

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/deit_tiny.npz")
ap.add_argument("--images", required=True)
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()

model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
sd = dict(np.load(a.weights))
# use the same binary16-rounded GEMM weights as the deployed model
for k in list(sd):
    if k.endswith(".weight") and sd[k].ndim >= 2:
        sd[k] = sd[k].astype(np.float16).astype(np.float32)
model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
model = model.double().eval()
w = Weights(a.weights)
worst = 0.0
for img, lab in image_source(a.images, 0, 0, a.n):
    with torch.no_grad():
        zt = model(torch.from_numpy(img[None]).double())[0].numpy()
    zn = ref_forward(w, img)["z"]
    d = float(np.max(np.abs(zt - zn)))
    worst = max(worst, d)
    print(f"max |logit diff| = {d:.3e}   argmax timm={zt.argmax()} numpy={zn.argmax()} label={lab}")
print("WORST", worst, "-> OK" if worst < 1e-6 else "-> MISMATCH, do not run the experiments")
