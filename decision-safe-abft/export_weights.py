"""Download DeiT-Tiny (ImageNet-1k) once and save its weights as NumPy arrays.

Needs PyTorch and timm only for this step:
    pip install torch timm
"""
import os
import numpy as np
import timm

os.makedirs("weights", exist_ok=True)
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
print("layer-norm eps:", model.blocks[0].norm1.eps, "(the code assumes 1e-6)")
sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
np.savez("weights/deit_tiny.npz", **sd)
print("saved weights/deit_tiny.npz with", len(sd), "arrays")
