"""Decision certificate relative to the fault-free ("golden") run.

The emulated accelerator is deterministic, so a fault that is not detected
can change the decision only through the difference it creates between the
faulty run (which we observe) and the golden run (which we do not). Radii are
per-token bounds on max_k |faulty - golden|. Upstream of the fault both runs
are identical and every radius is exactly zero; downstream, each emulated
unit contributes the proved propagation rule plus the rounding of both runs:

    |fl(F(x_f)) - fl(F(x_g))| <= S (1+u) + 2u |fl(F(x_f))| / (1-u),

where S bounds |F(x_f) - F(x_g)| and u is the unit's relative rounding.

Rows of the theta matrix (B x n_sites) are fault hypotheses; the paper uses
one row per site with that site's undetected-fault bound.
"""
import numpy as np

from .arith import U32, U16, gamma, max_abs_rows
from .model import D, H, DH, LN_EPS, STAGES

LG = 1.129            # > sup |GELU'| = Phi(sqrt2) + sqrt2*phi(sqrt2) = 1.12890...
SQRT_D = np.sqrt(D)
UR_UNIT = 2.0 * U32   # float64 unit rounded once to binary32 (covers float64 error)
TWO_M24 = 2.0 ** -24


def _two_run(S, out_inf, u):
    """Rounding of both runs on top of the exact-function difference S."""
    return np.where(S > 0, S * (1.0 + u) + 2.0 * u * out_inf / (1.0 - u), 0.0)


def _ln_rule(sd, eps, ginf):
    smin = np.sqrt(np.maximum(sd[None, :] - eps, 0.0) ** 2 + LN_EPS)
    return ginf * SQRT_D * eps / smin


def dist16(x):
    """Distance of each value to the nearest binary16 rounding boundary.
    If |x - y| is smaller than this distance, x and y round to the same
    binary16 number (round to nearest is constant between boundaries)."""
    x = np.asarray(x, np.float64)
    q16 = x.astype(np.float16)
    up = np.nextafter(q16, np.float16(np.inf)).astype(np.float64)
    dn = np.nextafter(q16, np.float16(-np.inf)).astype(np.float64)
    q = q16.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.minimum(np.abs(x - (q + up) / 2.0), np.abs(x - (q + dn) / 2.0))


def _count_le(sorted_d, eps):
    """Per token: how many input components lie within eps of a boundary."""
    out = np.empty(eps.shape)
    for t in range(eps.shape[1]):
        out[:, t] = np.searchsorted(sorted_d[t], eps[:, t], side="right")
    return out


def _gemm_two_run(eps_a, a_inf, a16_inf, Winf, Wmax, K, y_inf, theta, sorted_d):
    """Per-token difference of a binary16-input GEMM + bias between two runs.

    Only input components within eps of a binary16 rounding boundary can
    round differently in the two runs (n of them per token); the others are
    identical after the cast. If none can differ, the GEMM inputs are
    identical, the deterministic GEMM gives identical results, and the only
    difference is the fault itself. theta (B,1) bounds a fault added to the
    accumulator of this GEMM."""
    n = np.where(eps_a > 0, _count_le(sorted_d, eps_a), 0.0)
    D = eps_a * (1.0 + U16) + 2.0 * U16 * a_inf + TWO_M24
    lin = np.minimum(Winf, n * Wmax) * D
    acc = gamma(U32, K - 1) * Winf * (2.0 * a16_inf + D)
    eacc = np.where(n > 0, lin + acc, 0.0)
    return _two_run(eacc + theta, y_inf, U32)


class Stats:
    """Quantities of the observed run needed by the rules."""

    def __init__(self, w, run):
        self.w = w
        T = run["pre"]["x0"].shape[0]
        self.T = T
        self.blocks = []
        for b, rec in enumerate(run["blocks"]):
            blk = w.blocks[b]
            gm = rec["gemm"]
            s = {}
            s["sd_x"] = rec["x"].astype(np.float64).std(axis=1)
            s["a_inf"] = max_abs_rows(rec["a"])
            s["g1inf"] = float(np.max(np.abs(blk["g1"])))
            q = blk["qkv"]
            y = gm["qkv"]["y"]
            s["a16_qkv"] = max_abs_rows(gm["qkv"]["a16f"])
            s["sd_qkv"] = np.sort(dist16(rec["a"]), axis=1)
            for k, (lo, hi) in zip("qkv", ((0, D), (D, 2 * D), (2 * D, 3 * D))):
                s["inf_" + k] = q.slice_inf(lo, hi)
                s["max_" + k] = float(np.max(np.abs(q.W64[lo:hi])))
                s["y_" + k] = max_abs_rows(y[:, lo:hi])
            qh = y[:, :D].astype(np.float64).reshape(T, H, DH)
            kh = y[:, D:2 * D].astype(np.float64).reshape(T, H, DH)
            vh = y[:, 2 * D:].astype(np.float64).reshape(T, H, DH)
            s["qn1"] = np.abs(qh).sum(-1)
            s["kn1max"] = np.abs(kh).sum(-1).max(0)
            s["vrange"] = (vh.max(0) - vh.min(0)).max(-1)
            s["o_inf"] = max_abs_rows(rec["o"])
            for nm, inp in (("proj", rec["o"]), ("fc1", rec["h"]), ("fc2", rec["g"])):
                lin = blk[nm]
                s["inf_" + nm] = lin.inf
                s["max_" + nm] = float(np.max(np.abs(lin.W64)))
                s["sd_" + nm] = np.sort(dist16(inp), axis=1)
                s["K_" + nm] = lin.K
                s["in_" + nm] = max_abs_rows(inp)
                s["a16_" + nm] = max_abs_rows(gm[nm]["a16f"])
                s["y_" + nm] = max_abs_rows(gm[nm]["y"])
            s["x1_inf"] = max_abs_rows(rec["x1"])
            s["sd_x1"] = rec["x1"].astype(np.float64).std(axis=1)
            s["h_inf"] = max_abs_rows(rec["h"])
            s["g2inf"] = float(np.max(np.abs(blk["g2"])))
            s["g_inf"] = max_abs_rows(rec["g"])
            s["x2_inf"] = max_abs_rows(rec["x2"])
            s["K_qkv"] = q.K
            self.blocks.append(s)
        xL = run["blocks"][-1]["x2"][0].astype(np.float64)
        self.sd_cls = float(xL.std())
        self.gfinf = float(np.max(np.abs(w.gf)))
        hd = run["head"]
        self.c = hd["c"].astype(np.float64)
        self.c16 = hd["c16f"].astype(np.float64)
        self.c_inf = float(np.max(np.abs(self.c)))
        self.c_dist = dist16(self.c)
        self.z = hd["z"].astype(np.float64)
        self.Wabs_head = np.abs(w.head.W64)
        self.K_head = w.head.K
        self.c_hat = int(np.argmax(self.z)) if np.all(np.isfinite(self.z)) else -1


def propagate(st, theta, audit=None):
    """Return pass vector (B,) for the fault hypotheses in theta (B x n_sites).

    audit: optional dict(golden=run, run=run, row=int, log=list): check
    |faulty - golden| <= radius at every stage for one row.
    """
    theta = np.atleast_2d(np.asarray(theta, np.float64))
    B = theta.shape[0]
    T = st.T
    nb = len(st.blocks)
    eps = np.zeros((B, T))

    def chk(name, obs, gold, rad):
        if audit is None:
            return
        err = max_abs_rows(np.atleast_2d(np.asarray(obs, np.float64) - np.asarray(gold, np.float64)))
        r = np.atleast_1d(rad[audit["row"]])
        audit["log"].append((name, int(np.sum(err > r)), err.size))

    with np.errstate(over="ignore", invalid="ignore"):
        for b, s in enumerate(st.blocks):
            col = lambda stage: theta[:, 4 * b + STAGES.index(stage)][:, None]
            ea = _two_run(_ln_rule(s["sd_x"], eps, s["g1inf"]), s["a_inf"], UR_UNIT)
            tq = col("qkv")
            eq = _gemm_two_run(ea, s["a_inf"], s["a16_qkv"], s["inf_q"], s["max_q"], s["K_qkv"], s["y_q"], tq, s["sd_qkv"])
            ek = _gemm_two_run(ea, s["a_inf"], s["a16_qkv"], s["inf_k"], s["max_k"], s["K_qkv"], s["y_k"], tq, s["sd_qkv"])
            ev = _gemm_two_run(ea, s["a_inf"], s["a16_qkv"], s["inf_v"], s["max_v"], s["K_qkv"], s["y_v"], tq, s["sd_qkv"])
            Ek = ek.max(axis=1)
            Ev = ev.max(axis=1)
            eS = DH ** -0.5 * (s["qn1"][None] * Ek[:, None, None]
                               + eq[:, :, None] * (s["kn1max"][None, None, :] + DH * Ek[:, None, None]))
            So = (0.5 * eS * s["vrange"][None, None, :] + Ev[:, None, None]).max(axis=2)
            eo = _two_run(So, s["o_inf"], UR_UNIT)
            ep = _gemm_two_run(eo, s["in_proj"], s["a16_proj"], s["inf_proj"], s["max_proj"], s["K_proj"], s["y_proj"],
                               col("proj"), s["sd_proj"])
            ex1 = _two_run(eps + ep, s["x1_inf"], U32)
            eh = _two_run(_ln_rule(s["sd_x1"], ex1, s["g2inf"]), s["h_inf"], UR_UNIT)
            ef1 = _gemm_two_run(eh, s["in_fc1"], s["a16_fc1"], s["inf_fc1"], s["max_fc1"], s["K_fc1"], s["y_fc1"],
                                col("fc1"), s["sd_fc1"])
            eg = _two_run(LG * ef1, s["g_inf"], UR_UNIT)
            ef2 = _gemm_two_run(eg, s["in_fc2"], s["a16_fc2"], s["inf_fc2"], s["max_fc2"], s["K_fc2"], s["y_fc2"],
                                col("fc2"), s["sd_fc2"])
            ex2 = _two_run(ex1 + ef2, s["x2_inf"], U32)
            if audit is not None:
                rf, rg = audit["run"]["blocks"][b], audit["golden"]["blocks"][b]
                yf, yg = rf["gemm"]["qkv"]["y"], rg["gemm"]["qkv"]["y"]
                chk(f"b{b}.a", rf["a"], rg["a"], ea)
                chk(f"b{b}.q", yf[:, :D], yg[:, :D], eq)
                chk(f"b{b}.k", yf[:, D:2 * D], yg[:, D:2 * D], ek)
                chk(f"b{b}.v", yf[:, 2 * D:], yg[:, 2 * D:], ev)
                chk(f"b{b}.o", rf["o"], rg["o"], eo)
                chk(f"b{b}.proj", rf["gemm"]["proj"]["y"], rg["gemm"]["proj"]["y"], ep)
                chk(f"b{b}.x1", rf["x1"], rg["x1"], ex1)
                chk(f"b{b}.h", rf["h"], rg["h"], eh)
                chk(f"b{b}.fc1", rf["gemm"]["fc1"]["y"], rg["gemm"]["fc1"]["y"], ef1)
                chk(f"b{b}.g", rf["g"], rg["g"], eg)
                chk(f"b{b}.fc2", rf["gemm"]["fc2"]["y"], rg["gemm"]["fc2"]["y"], ef2)
                chk(f"b{b}.x2", rf["x2"], rg["x2"], ex2)
            eps = ex2
        # final norm on the class token
        e_cls = eps[:, 0]
        smin = np.sqrt(np.maximum(st.sd_cls - e_cls, 0.0) ** 2 + LN_EPS)
        ec = _two_run(st.gfinf * SQRT_D * e_cls / smin, st.c_inf, UR_UNIT)       # (B,)
        # head GEMM, componentwise per class
        mask = (ec[:, None] > 0) & (st.c_dist[None, :] <= ec[:, None])   # may round differently
        d16 = np.where(mask, ec[:, None] * (1.0 + U16) + 2.0 * U16 * np.abs(st.c)[None, :] + TWO_M24, 0.0)
        anym = mask.any(axis=1, keepdims=True)
        eacc = d16 @ st.Wabs_head.T + gamma(U32, st.K_head - 1) * (
            (2.0 * np.abs(st.c16)[None, :] + d16) @ st.Wabs_head.T)
        eacc = np.where(anym, eacc, 0.0)
        th = theta[:, nb * 4][:, None]                     # head site
        zabs = np.abs(st.z)[None, :]
        ez0 = _two_run(eacc, zabs, U32)
        ezp = _two_run(eacc + th, zabs, U32)
    if audit is not None:
        chk("c", audit["run"]["head"]["c"][None], audit["golden"]["head"]["c"][None], ec)
        zerr = np.abs(st.z - audit["golden"]["head"]["z"].astype(np.float64))
        bound = np.maximum(ez0, ezp)[audit["row"]]
        audit["log"].append(("z", int(np.sum(zerr > bound)), zerr.size))
    c = st.c_hat
    if c < 0:
        return np.zeros(B, dtype=bool)
    gap = st.z[c] - st.z                                    # (C,)
    # a single-entry head fault touches at most one of the two classes compared
    need = np.maximum(ezp[:, c:c + 1] + ez0, ez0[:, c:c + 1] + ezp)
    others = np.arange(need.shape[1]) != c
    with np.errstate(invalid="ignore"):
        ok = np.all(gap[None, others] > need[:, others], axis=1)
    return ok


def single_site_pass(st, site, theta_value, audit=None):
    """Certificate for an undetected fault at a known site (window theorem per tile)."""
    th = np.zeros((1, len(st.w.sites)))
    th[0, site] = theta_value
    return bool(propagate(st, th, audit)[0])


def theta_star(st, n_sites, lo=-10.0, hi=4.0, n1=113, n2=33):
    """Largest certifiable fault bound per site (grid search; the certificate
    is monotone in theta). 0 where nothing above 10^lo certifies."""
    g1 = np.logspace(lo, hi, n1)
    Th = np.zeros((n_sites * n1, n_sites))
    for s in range(n_sites):
        Th[s * n1:(s + 1) * n1, s] = g1
    ok = propagate(st, Th).reshape(n_sites, n1)
    out = np.zeros(n_sites)
    brk = {}
    for s in range(n_sites):
        idx = np.where(ok[s])[0]
        if idx.size == 0:
            continue
        out[s] = g1[idx[-1]]
        if idx[-1] < n1 - 1:
            brk[s] = (g1[idx[-1]], g1[idx[-1] + 1])
    if brk:
        keys = list(brk)
        Th2 = np.zeros((len(keys) * n2, n_sites))
        grids = {}
        for r, s in enumerate(keys):
            grids[s] = np.logspace(np.log10(brk[s][0]), np.log10(brk[s][1]), n2)
            Th2[r * n2:(r + 1) * n2, s] = grids[s]
        ok2 = propagate(st, Th2).reshape(len(keys), n2)
        for r, s in enumerate(keys):
            idx = np.where(ok2[r])[0]
            if idx.size:
                out[s] = grids[s][idx[-1]]
    return out
