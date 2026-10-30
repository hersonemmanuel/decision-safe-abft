"""Create random DeiT-Tiny-shaped weights for the self-test (no download needed)."""
import sys
import numpy as np

def main(path="weights/synthetic.npz", depth=12, seed=0):
    r = np.random.default_rng(seed)
    D, MLP = 192, 768
    p = {"cls_token": r.normal(0, .02, (1, 1, D)), "pos_embed": r.normal(0, .02, (1, 197, D)),
         "patch_embed.proj.weight": r.normal(0, .02, (D, 3, 16, 16)), "patch_embed.proj.bias": np.zeros(D),
         "norm.weight": np.ones(D), "norm.bias": np.zeros(D),
         "head.weight": r.normal(0, .1, (1000, D)), "head.bias": np.zeros(1000)}
    for b in range(depth):
        q = f"blocks.{b}."
        p.update({q + "norm1.weight": 1 + r.normal(0, .1, D), q + "norm1.bias": r.normal(0, .02, D),
                  q + "attn.qkv.weight": r.normal(0, .05, (3 * D, D)), q + "attn.qkv.bias": r.normal(0, .02, 3 * D),
                  q + "attn.proj.weight": r.normal(0, .05, (D, D)), q + "attn.proj.bias": np.zeros(D),
                  q + "norm2.weight": 1 + r.normal(0, .1, D), q + "norm2.bias": np.zeros(D),
                  q + "mlp.fc1.weight": r.normal(0, .05, (MLP, D)), q + "mlp.fc1.bias": np.zeros(MLP),
                  q + "mlp.fc2.weight": r.normal(0, .03, (D, MLP)), q + "mlp.fc2.bias": np.zeros(D)})
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: v.astype(np.float32) for k, v in p.items()})
    print("wrote", path)

if __name__ == "__main__":
    main(*sys.argv[1:2])
