"""E1: is the fault-free residual bound sound, and how tight is it?

Tiles are cut from real activations and weights of the model. For each tile
the main product, the column sum and the checksum path are re-computed with
an explicit accumulation order (sequential or binary tree) and an explicit
accumulator format (binary32 or bfloat16), so every rounding is known. The
observed residual is compared with the accumulator-aware bound.
An INT8 version of each tile checks that the integer residual is exactly 0.
"""
import argparse
import csv
import os

import numpy as np

from dcft.arith import U32, UBF, gamma, round_bf16
from dcft.model import Weights, hw_forward, STAGES
from common import image_source

CONFIGS = [("fp32", "seq"), ("fp32", "tree"), ("bf16", "seq"), ("bf16", "tree")]


def reduce_seq(X, rnd):
    acc = X[0].copy()
    for i in range(1, X.shape[0]):
        acc = rnd(acc + X[i])
    return acc


def reduce_tree(X, rnd):
    n = X.shape[0]
    p = 1 << int(np.ceil(np.log2(n))) if n > 1 else 1
    if p > n:
        pad = np.zeros((p - n,) + X.shape[1:], dtype=X.dtype)
        X = np.concatenate([X, pad], 0)
    while X.shape[0] > 1:
        X = rnd(X[0::2] + X[1::2])
    return X[0]


def emulate(A, B, fmt, order):
    """A (m,K), B (K,n): binary16 values stored as float32. Returns sigma, t (float64)."""
    rnd = (lambda v: v.astype(np.float32)) if fmt == "fp32" else round_bf16
    red = reduce_seq if order == "seq" else reduce_tree
    P = (A[:, :, None] * B[None, :, :]).astype(np.float32)        # exact products
    C = red(np.moveaxis(P, 1, 0), rnd)                              # (m, n)
    sigma = red(C, rnd)                                             # (n,)
    w = red(A, rnd)                                                 # (K,)
    prod = (w[:, None] * B).astype(np.float32)                      # RN in fp32
    if fmt == "bf16":
        # bf16 x fp16 has at most 19 significant bits: exact in fp32
        assert np.array_equal(prod.astype(np.float64), w.astype(np.float64)[:, None] * B.astype(np.float64))
    t = red(prod, rnd)
    return sigma.astype(np.float64), t.astype(np.float64)


def eta_coeff(fmt, order, m, K):
    u = U32 if fmt == "fp32" else UBF
    up = U32 if fmt == "fp32" else 0.0
    if order == "seq":
        dC, dS, dw, dt = K - 1, m - 1, m - 1, K - 1
    else:
        L = lambda n: int(np.ceil(np.log2(n)))
        dC, dS, dw, dt = L(K), L(m), L(m), L(K)
    main = gamma(u, dC + dS)
    refp = (1 + u) ** (dw + dt) * (1 + up) - 1
    return main + refp


def int8_check(A, B):
    sa = np.max(np.abs(A)) / 127.0 or 1.0
    sb = np.max(np.abs(B)) / 127.0 or 1.0
    Ai = np.clip(np.round(A / sa), -128, 127).astype(np.int64)
    Bi = np.clip(np.round(B / sb), -128, 127).astype(np.int64)
    C = Ai @ Bi
    sig = C.sum(0)
    t = Ai.sum(0) @ Bi
    biggest = max(np.abs(C).max(), np.abs(sig).max(), np.abs(t).max(),
                  np.abs(np.cumsum(Ai[:, :, None] * Bi[None], axis=1)).max())
    return int(np.max(np.abs(sig - t))), int(biggest) < 2 ** 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--tiles-per-image", type=int, default=6)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    w = Weights(a.weights)
    rows, ratios = [], {c: [] for c in CONFIGS}
    int8_zero, int8_total, int8_fit = 0, 0, 0
    for idx, (img, _) in enumerate(image_source(a.images, a.synthetic, 0, a.n_images)):
        run = hw_forward(w, img)
        for _ in range(a.tiles_per_image):
            b = int(rng.integers(w.depth))
            stage = STAGES[int(rng.integers(4))]
            g = run["blocks"][b]["gemm"][stage]
            lin = w.blocks[b][stage]
            Afull = g["a16f"]
            rsel = rng.choice(Afull.shape[0], size=min(a.m, Afull.shape[0]), replace=False)
            csel = rng.choice(lin.N, size=min(a.n, lin.N), replace=False)
            A = Afull[np.sort(rsel)]
            B = lin.W32T[:, np.sort(csel)]
            m, K = A.shape
            omega = np.abs(A.astype(np.float64)).sum(0) @ np.abs(B.astype(np.float64))
            for cfg in CONFIGS:
                sig, t = emulate(A, B, *cfg)
                r = np.abs(sig - t)
                coeff = eta_coeff(*cfg, m, K)
                eta = coeff * omega
                rat = r / eta
                ratios[cfg].append(rat)
                uu = U32 if cfg[0] == "fp32" else 2.0 ** -8
                # heuristic thresholds of the kind used when no sound bound is available
                tau_lin = K * uu * omega          # leading term of the classical bound
                tau_sqrt = np.sqrt(K) * uu * omega  # leading term of the probabilistic bound
                tau_rel = K * uu * np.abs(t)        # relative to the checksum value itself
                rows.append(dict(image=idx, block=b, stage=stage, fmt=cfg[0], order=cfg[1],
                                 m=m, K=K, coeff=coeff, classical=(K + m) * uu,
                                 classical_defined=bool((K + m) * uu < 1),
                                 max_ratio=float(rat.max()), median_ratio=float(np.median(rat)),
                                 violations=int(np.sum(r > eta)), cols=len(r),
                                 fa_lin=int(np.sum(r > tau_lin)), fa_sqrt=int(np.sum(r > tau_sqrt)),
                                 fa_rel=int(np.sum(r > tau_rel))))
            z, fit = int8_check(A.astype(np.float64), B.astype(np.float64))
            int8_total += 1
            int8_zero += int(z == 0)
            int8_fit += int(fit)
        print(f"E1 image {idx + 1}/{a.n_images} done", flush=True)
    with open(os.path.join(a.out, "e1_tiles.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(os.path.join(a.out, "e1_ratios.npz"),
             **{f"{c[0]}_{c[1]}": np.concatenate(ratios[c]) for c in CONFIGS},
             int8=np.array([int8_zero, int8_total, int8_fit]))
    print("E1 finished. int8 zero-residual tiles:", int8_zero, "/", int8_total)


if __name__ == "__main__":
    main()
