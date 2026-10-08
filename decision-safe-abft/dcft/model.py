"""DeiT-Tiny in NumPy: an exact-arithmetic reference and an emulated accelerator.

Reference: float64 evaluation of the deployed model (GEMM weights rounded to
binary16, everything else as stored). It stands in for exact arithmetic; its
own error is around 1e-15 relative and far below every bound we test.

Emulated accelerator ("hardware"):
  * weight GEMMs: binary16 inputs and weights, products exact, binary32
    accumulation by the platform BLAS (order unknown, so bounds use the
    worst-case depth K-1); bias added in binary32;
  * the 48 weight GEMMs inside the blocks are checked with column checksums;
  * layer norm, attention core, GELU: computed in float64 and rounded once
    to binary32 (a high-precision unit with a final rounding);
  * residual additions in binary32.
"""
import numpy as np
from scipy.special import erf

from .arith import (U32, U16, F16_HALF_SUB, F32_TINY, gamma, flip_bit_f32,
                    max_abs_rows)

D, H, DH, MLP, NCLS = 192, 3, 64, 768, 1000
LN_EPS = 1e-6
STAGES = ("qkv", "proj", "fc1", "fc2")


class Linear:
    """A weight GEMM y = W x + b with W deployed in binary16."""

    def __init__(self, W, b):
        W16 = np.asarray(W, dtype=np.float32).astype(np.float16)
        if not np.all(np.isfinite(W16)):
            raise ValueError("weight overflows binary16")
        self.W64 = W16.astype(np.float64)              # (N, K) deployed weight
        self.W32T = np.ascontiguousarray(W16.astype(np.float32).T)   # (K, N)
        self.absW32T = np.abs(self.W32T)
        self.b32 = np.asarray(b, dtype=np.float32)
        self.b64 = self.b32.astype(np.float64)
        self.N, self.K = self.W64.shape
        self.row_l1 = np.abs(self.W64).sum(axis=1)       # per output, sum_k |W|
        self.inf = float(self.row_l1.max())               # ||W||_inf

    def slice_inf(self, lo, hi):
        return float(self.row_l1[lo:hi].max())


class Weights:
    def __init__(self, npz_path):
        p = dict(np.load(npz_path))
        self.depth = 1 + max(int(k.split(".")[1]) for k in p if k.startswith("blocks."))
        f32 = lambda k: np.asarray(p[k], dtype=np.float32)
        self.patch = Linear(f32("patch_embed.proj.weight").reshape(D, -1),
                            f32("patch_embed.proj.bias"))
        self.cls32 = f32("cls_token").reshape(D)
        self.pos32 = f32("pos_embed").reshape(-1, D)
        self.blocks = []
        for b in range(self.depth):
            q = f"blocks.{b}."
            self.blocks.append(dict(
                g1=f32(q + "norm1.weight"), be1=f32(q + "norm1.bias"),
                qkv=Linear(f32(q + "attn.qkv.weight"), f32(q + "attn.qkv.bias")),
                proj=Linear(f32(q + "attn.proj.weight"), f32(q + "attn.proj.bias")),
                g2=f32(q + "norm2.weight"), be2=f32(q + "norm2.bias"),
                fc1=Linear(f32(q + "mlp.fc1.weight"), f32(q + "mlp.fc1.bias")),
                fc2=Linear(f32(q + "mlp.fc2.weight"), f32(q + "mlp.fc2.bias")),
            ))
        self.gf, self.bf = f32("norm.weight"), f32("norm.bias")
        self.head = Linear(f32("head.weight"), f32("head.bias"))
        self.sites = [(b, s) for b in range(self.depth) for s in STAGES] + [(self.depth, "head")]

    def lin(self, site):
        b, s = site
        return self.head if s == "head" else self.blocks[b][s]


# ----------------------------------------------------------------------
# shared exact-function pieces (float64)
# ----------------------------------------------------------------------
def patchify(img):
    """(3,224,224) -> (196, 768) in the order (C, kh, kw), row-major patches."""
    x = np.asarray(img).reshape(3, 14, 16, 14, 16)
    return x.transpose(1, 3, 0, 2, 4).reshape(196, 768)


def ln64(x, g, b):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * np.asarray(g, np.float64) + np.asarray(b, np.float64)


def gelu64(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def attn64(qkv):
    """qkv: (T, 3D) -> attention output (T, D), exact softmax attention."""
    qkv = np.asarray(qkv, dtype=np.float64)
    T = qkv.shape[0]
    q = qkv[:, :D].reshape(T, H, DH).transpose(1, 0, 2)
    k = qkv[:, D:2 * D].reshape(T, H, DH).transpose(1, 0, 2)
    v = qkv[:, 2 * D:].reshape(T, H, DH).transpose(1, 0, 2)
    s = (q * DH ** -0.5) @ k.transpose(0, 2, 1)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return (p @ v).transpose(1, 0, 2).reshape(T, D)


# ----------------------------------------------------------------------
# reference forward (float64)
# ----------------------------------------------------------------------
def ref_forward(w, img):
    x = patchify(np.asarray(img, np.float64)) @ w.patch.W64.T + w.patch.b64
    x = np.concatenate([w.cls32.astype(np.float64)[None], x], 0) + w.pos32.astype(np.float64)
    tr = {"x0": x, "blocks": []}
    for blk in w.blocks:
        a = ln64(x, blk["g1"], blk["be1"])
        qkv = a @ blk["qkv"].W64.T + blk["qkv"].b64
        o = attn64(qkv)
        p = o @ blk["proj"].W64.T + blk["proj"].b64
        x1 = x + p
        h = ln64(x1, blk["g2"], blk["be2"])
        f1 = h @ blk["fc1"].W64.T + blk["fc1"].b64
        g = gelu64(f1)
        f2 = g @ blk["fc2"].W64.T + blk["fc2"].b64
        x2 = x1 + f2
        tr["blocks"].append(dict(x=x, a=a, qkv=qkv, o=o, proj=p, x1=x1, h=h,
                                 fc1=f1, g=g, fc2=f2, x2=x2))
        x = x2
    c = ln64(x[0:1], w.gf, w.bf)[0]
    z = c @ w.head.W64.T + w.head.b64
    tr.update(c=c, z=z)
    return tr


# ----------------------------------------------------------------------
# emulated accelerator
# ----------------------------------------------------------------------
def checked_gemm(lin, a32, fault=None):
    """binary16-input GEMM with binary32 accumulation and a column checksum.

    fault: None or (row, col, bit) -> invert that bit of the binary32
    accumulator entry before the check.
    """
    a16 = a32.astype(np.float16)
    a16f = a16.astype(np.float32)
    acc = a16f @ lin.W32T
    if fault is not None:
        i, j, bit = fault
        acc[i, j] = flip_bit_f32(acc[i, j], bit)
    nonfinite = (not np.all(np.isfinite(a16f))) or (not np.all(np.isfinite(acc)))
    with np.errstate(over="ignore", invalid="ignore"):
        sig = acc.sum(axis=0, dtype=np.float32)
        wsum = a16f.sum(axis=0, dtype=np.float32)
        t = (wsum[None, :] @ lin.W32T)[0]
        r = (sig - t).astype(np.float32)
        omega_hat = (np.abs(a16f).sum(axis=0, dtype=np.float32)[None, :] @ lin.absW32T)[0]
        y = acc + lin.b32
    return dict(a16f=a16f, acc=acc, y=y, r=r, omega_hat=omega_hat,
                nonfinite=nonfinite or (not np.all(np.isfinite(r))))


def eta_bound(omega_hat, m, K):
    """Sound fault-free residual bound for the column checksum (paper Thm. 6).

    Worst-case depths because the BLAS order is unknown:
    main product K-1, column sum m-1, operand checksum m-1, checksum
    product 1, reference sum K-1. The computed weight is divided by
    (1-u)^d to give an upper bound on the exact weight.
    """
    coeff = gamma(U32, (K - 1) + (m - 1)) + gamma(U32, (m - 1) + 1 + (K - 1))
    d_om = (m - 1) + 1 + (K - 1)
    omega_up = np.asarray(omega_hat, np.float64) / (1.0 - U32) ** d_om
    return coeff * omega_up


def gamma_sigma(m):
    return gamma(U32, m - 1)


def thresholds(eta, theta_site, policy, m):
    """Per-column thresholds and the resulting undetected-fault bound theta.

    policy 'P0': rounding-only, tau = (1+u) eta.
    policy 'P1'/'P2': certified window, tau as large as theta_site allows.
    Returns (tau, theta_actual) where theta_actual = max_j bound on an
    undetected single-entry fault (paper Thm. 7).
    """
    Gs = gamma_sigma(m)
    lo = (1.0 + U32) * eta
    if policy == "P0" or theta_site is None:
        tau = lo
    else:
        hi = (1.0 - U32) * ((1.0 - Gs) * theta_site - eta)
        tau = np.maximum(lo, hi)
    theta_cols = (tau / (1.0 - U32) + eta) / (1.0 - Gs)
    return tau, float(theta_cols.max())


def theta_observed(r, eta, m):
    """A posteriori bound on an undetected single-entry fault (paper Cor. 3):
    the observed residual itself bounds the fault, so a small fault gets a
    small bound. The faulty column is unknown, hence the max over columns."""
    Gs = gamma_sigma(m)
    return float(np.max((np.abs(np.asarray(r, np.float64)) / (1.0 - U32) + eta) / (1.0 - Gs)))


def hw_patch_stage(w, img32):
    lin = w.patch
    pts = patchify(np.asarray(img32, np.float32))
    a16f = pts.astype(np.float16).astype(np.float32)
    acc = a16f @ lin.W32T
    y = acc + lin.b32
    x = np.concatenate([w.cls32[None], y], 0) + w.pos32
    return dict(x0=x.astype(np.float32), pts=pts, y=y)


def hw_blocks(w, x32, b_start=0, fault=None):
    """Run blocks b_start..end. fault: None or (block, stage, row, col, bit)."""
    blocks = []
    x = x32
    for b in range(b_start, w.depth):
        blk = w.blocks[b]
        rec = {"x": x}
        ft = lambda st: (fault[2], fault[3], fault[4]) if (fault is not None and fault[0] == b and fault[1] == st) else None
        a = ln64(x, blk["g1"], blk["be1"]).astype(np.float32)
        rec["a"] = a
        g_qkv = checked_gemm(blk["qkv"], a, ft("qkv"))
        o = attn64(g_qkv["y"]).astype(np.float32)
        g_proj = checked_gemm(blk["proj"], o, ft("proj"))
        x1 = (x + g_proj["y"]).astype(np.float32)
        hh = ln64(x1, blk["g2"], blk["be2"]).astype(np.float32)
        g_fc1 = checked_gemm(blk["fc1"], hh, ft("fc1"))
        g = gelu64(g_fc1["y"]).astype(np.float32)
        g_fc2 = checked_gemm(blk["fc2"], g, ft("fc2"))
        x2 = (x1 + g_fc2["y"]).astype(np.float32)
        rec.update(o=o, x1=x1, h=hh, g=g, x2=x2,
                   gemm=dict(qkv=g_qkv, proj=g_proj, fc1=g_fc1, fc2=g_fc2))
        blocks.append(rec)
        x = x2
    return blocks, x


def hw_head(w, x32, fault=None):
    """Final norm (cls token) and the classifier head, checked like the others
    (m = 1 row, so the column check amounts to a recomputation)."""
    c = ln64(x32[0:1], w.gf, w.bf).astype(np.float32)
    g = checked_gemm(w.head, c, fault)
    return dict(c=c[0], c16f=g["a16f"][0], z=g["y"][0], gemm=g)


def hw_forward(w, img32):
    pre = hw_patch_stage(w, img32)
    blocks, xL = hw_blocks(w, pre["x0"], 0, None)
    head = hw_head(w, xL)
    return dict(pre=pre, blocks=blocks, head=head)


def hw_forward_faulty(w, clean, fault):
    """Re-run from the faulty block using the stored clean block input.
    fault = (block, stage, row, col, bit); block == w.depth means the head."""
    b0 = fault[0]
    if b0 == w.depth:
        head = hw_head(w, clean["blocks"][-1]["x2"], (fault[2], fault[3], fault[4]))
        return dict(pre=clean["pre"], blocks=clean["blocks"], head=head)
    blocks, xL = hw_blocks(w, clean["blocks"][b0]["x"], b0, fault)
    head = hw_head(w, xL)
    return dict(pre=clean["pre"], blocks=clean["blocks"][:b0] + blocks, head=head)


def site_gemm(w, run, site):
    b, s = site
    return run["head"]["gemm"] if s == "head" else run["blocks"][b]["gemm"][s]
