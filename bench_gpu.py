"""End-to-end cost on a GPU: inference, checks, certificate, memory, and the
expected cost of recomputation under a given fault rate.

Parts
  A  inference latency (batch 1) and throughput (batch B) with PyTorch:
       fp16      half-precision deployment baseline (tensor cores)
       emu       the arithmetic of the paper: binary16 GEMM inputs, binary32
                 accumulation (SGEMM, TF32 off), binary32 elsewhere
     and the same emulated model with checks on all 49 GEMMs:
       col       one column checksum over all rows (the baseline ABFT)
       grouped   class token on its own + one group for the other rows
       grouped16 class token + groups of 16 rows
       twosided  grouped + one row checksum per row
     plus peak GPU memory for each.
  B  certificate cost per inference at the sites enabled by E2b, on the CPU
     (NumPy) and on the GPU (CuPy, if installed), and agreement of the two.
  C  expected time per inference under a fault rate lambda (faults per
     inference), using the recomputation probabilities measured in E3b:
       T = T_inference + T_checks + T_certificate + lambda * p_recompute * T_inference

Run:
  python bench_gpu.py --weights weights/deit_tiny.npz --images "$IMAGES" \
      --calib results_v5/calib_v5.json --e3b results_v5/e3b_injections.csv
Results go to results_v5/bench_gpu.json and are printed as a table.
"""
import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from dcft.arith import U32, gamma
from dcft.model import Weights
from dcft.faultsim import hw_forward_g, gemm_of, theta_rows, eta_coeff
from common import image_source

D, H, DH, MLP, LN_EPS = 192, 3, 64, 768, 1e-6


# ----------------------------------------------------------------------
# A: PyTorch model
# ----------------------------------------------------------------------
def torch_model(npz, dev):
    import torch
    p = dict(np.load(npz))
    t = lambda k, dt=torch.float32: torch.tensor(p[k], dtype=dt, device=dev)
    depth = 1 + max(int(k.split(".")[1]) for k in p if k.startswith("blocks."))

    def lin(name):
        W = torch.tensor(p[name + ".weight"].reshape(p[name + ".weight"].shape[0], -1),
                         dtype=torch.float32).half()
        b = torch.tensor(p[name + ".bias"], dtype=torch.float32)
        W16 = W.to(dev)
        W32T = W16.float().t().contiguous()
        return dict(W16=W16, b16=b.half().to(dev), W32T=W32T, absW32T=W32T.abs(), b32=b.to(dev),
                    wrow=W32T.sum(1), absrow=W32T.abs().sum(1), K=W32T.shape[0], N=W32T.shape[1])
    m = dict(depth=depth, patch=lin("patch_embed.proj"), cls=t("cls_token").reshape(1, 1, D),
             pos=t("pos_embed").reshape(1, -1, D), gf=t("norm.weight"), bf=t("norm.bias"),
             head=lin("head"), blocks=[])
    for b in range(depth):
        q = f"blocks.{b}."
        m["blocks"].append(dict(g1=t(q + "norm1.weight"), be1=t(q + "norm1.bias"),
                                g2=t(q + "norm2.weight"), be2=t(q + "norm2.bias"),
                                qkv=lin(q + "attn.qkv"), proj=lin(q + "attn.proj"),
                                fc1=lin(q + "mlp.fc1"), fc2=lin(q + "mlp.fc2")))
    return m


def patchify_t(x):
    B = x.shape[0]
    return x.reshape(B, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(B, 196, 768)


class Checker:
    """Column / grouped / two-sided checks on a batched GEMM (B, T, N).

    Implementation for speed, with the same soundness as the NumPy emulator:
      * all group sums of a GEMM are formed at once (rows padded with zeros
        to whole groups; zero rows add nothing to any sum), and t and omega
        come from one matrix product with [W | |W|];
      * the threshold (1+u) eta = c * omega_hat is evaluated in binary32 with
        a constant c rounded up so that the binary32 product can only be
        larger than the exact threshold (c includes a factor (1 + 4u));
        the residual itself is the binary32 value r = fl(sigma - t) that the
        theory models (Theorem 4). A group smaller than the nominal size
        (the last one) uses the constant of the nominal size, which is larger.
    """

    def __init__(self, mode, gsize=16):
        self.mode, self.gsize = mode, gsize
        self.alarm = None
        self.cache = {}

    def reset(self, B, dev):
        import torch
        self.alarm = torch.zeros(B, dtype=torch.bool, device=dev)

    @staticmethod
    def _c(m, K):
        u = U32
        return (1.0 + u) * eta_coeff(m, K) / (1.0 - u) ** ((m - 1) + 1 + (K - 1)) * (1.0 + 4.0 * u)

    def _setup(self, L, T):
        import torch
        key = (id(L), T)
        if key in self.cache:
            return self.cache[key]
        K, N = L["K"], L["N"]
        if self.mode == "col" or T == 1:
            sizes, g = [T], None
        elif self.mode in ("grouped", "twosided"):
            sizes, g = [1, T - 1], T - 1
        else:
            g = self.gsize
            nG = -(-(T - 1) // g)
            sizes = [1] + [g] * nG
        c = torch.tensor([self._c(m, K) for m in sizes], device=L["W32T"].device,
                         dtype=torch.float32).reshape(1, -1, 1)
        Wcat = torch.cat([L["W32T"], L["absW32T"]], 1)          # (K, 2N)
        cr = None
        if self.mode == "twosided":
            Wr = torch.zeros(2 * K, 2, device=L["W32T"].device)
            Wr[:K, 0], Wr[K:, 1] = L["wrow"], L["absrow"]
            cr = (Wr, self._c(N, K))
        self.cache[key] = (sizes, g, c, Wcat, cr)
        return self.cache[key]

    def __call__(self, a16, acc, L):
        if self.mode == "none":
            return
        import torch
        B, T, N = acc.shape
        sizes, g, c, Wcat, cr = self._setup(L, T)
        if len(sizes) == 1:
            sa, sx = acc.sum(1, keepdim=True), a16.sum(1, keepdim=True)
            sabs = a16.abs().sum(1, keepdim=True)
        else:
            R = T - 1
            nG = len(sizes) - 1
            pad = nG * g - R
            ar, xr = acc[:, 1:], a16[:, 1:]
            if pad:
                ar = torch.nn.functional.pad(ar, (0, 0, 0, pad))
                xr = torch.nn.functional.pad(xr, (0, 0, 0, pad))
            sa = torch.cat([acc[:, :1], ar.reshape(B, nG, g, N).sum(2)], 1)
            xg = xr.reshape(B, nG, g, -1)
            sx = torch.cat([a16[:, :1], xg.sum(2)], 1)
            sabs = torch.cat([a16[:, :1].abs(), xg.abs().sum(2)], 1)
        G = sa.shape[1]
        tw = torch.cat([sx, sabs], 1) @ Wcat                    # (B, 2G, 2N)
        t, om = tw[:, :G, :N], tw[:, G:, N:]
        r = sa - t
        bad = (r.abs() > c * om) | ~torch.isfinite(r)
        self.alarm |= bad.flatten(1).any(1)
        if cr is not None:
            Wr, crow = cr
            sr = acc.sum(2)
            tr = torch.cat([a16, a16.abs()], 2) @ Wr               # (B, T, 2)
            rr = sr - tr[..., 0]
            self.alarm |= ((rr.abs() > crow * tr[..., 1]) | ~torch.isfinite(rr)).any(1)


def forward_emu(m, x, chk):
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    chk.reset(B, x.device)

    def gemm(a, L):
        a16 = a.half().float()
        acc = a16 @ L["W32T"]
        chk(a16, acc, L)
        return acc + L["b32"]
    pts = patchify_t(x)
    y = pts.half().float() @ m["patch"]["W32T"] + m["patch"]["b32"]
    h = torch.cat([m["cls"].expand(B, 1, D), y], 1) + m["pos"]
    T = h.shape[1]
    for blk in m["blocks"]:
        a = F.layer_norm(h, (D,), blk["g1"], blk["be1"], LN_EPS)
        qkv = gemm(a, blk["qkv"]).reshape(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)
        s = (qkv[0] * DH ** -0.5) @ qkv[1].transpose(-1, -2)
        o = (torch.softmax(s, -1) @ qkv[2]).transpose(1, 2).reshape(B, T, D)
        h = h + gemm(o, blk["proj"])
        g = F.gelu(gemm(F.layer_norm(h, (D,), blk["g2"], blk["be2"], LN_EPS), blk["fc1"]))
        h = h + gemm(g, blk["fc2"])
    c = F.layer_norm(h[:, :1], (D,), m["gf"], m["bf"], LN_EPS)
    return gemm(c, m["head"])[:, 0]


def forward_fp16(m, x):
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    lin = lambda a, L: F.linear(a, L["W16"], L["b16"])
    pts = patchify_t(x).half()
    h = torch.cat([m["cls"].half().expand(B, 1, D), lin(pts, m["patch"])], 1) + m["pos"].half()
    T = h.shape[1]
    for blk in m["blocks"]:
        a = F.layer_norm(h, (D,), blk["g1"].half(), blk["be1"].half(), LN_EPS)
        qkv = lin(a, blk["qkv"]).reshape(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, T, D)
        h = h + lin(o, blk["proj"])
        h = h + lin(F.gelu(lin(F.layer_norm(h, (D,), blk["g2"].half(), blk["be2"].half(), LN_EPS),
                                blk["fc1"])), blk["fc2"])
    c = F.layer_norm(h[:, :1], (D,), m["gf"].half(), m["bf"].half(), LN_EPS)
    return lin(c, m["head"])[:, 0]


def time_gpu(fn, reps, warm=10, graph=False):
    """Median and 90th percentile time in ms. graph=True captures fn in a
    CUDA graph once and times its replay (removes kernel-launch overhead)."""
    import torch
    run = fn
    if graph:
        try:
            st_ = torch.cuda.Stream()
            st_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st_):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(st_)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            run = g.replay
        except Exception as e:
            print("    (CUDA graph capture failed, eager timing:", repr(e), ")")
            return None, None
    for _ in range(warm):
        run()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        st.record()
        run()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    return float(np.median(ts)), float(np.percentile(ts, 90))


def part_A(a, imgs, out):
    import torch
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = torch_model(a.weights, dev)
    X = torch.tensor(np.stack(imgs), device=dev)
    res = {}
    with torch.no_grad():
        for B in (1, a.batch):
            x = X[:B] if B <= X.shape[0] else X.repeat((B + X.shape[0] - 1) // X.shape[0], 1, 1, 1)[:B]
            reps = a.reps if B == 1 else max(10, a.reps // 4)
            cfgs = [("fp16", lambda: forward_fp16(m, x))]
            for mode in ("none", "col", "grouped", "grouped16", "twosided"):
                ck = Checker(mode)
                cfgs.append((f"emu+{mode}", (lambda ck=ck: forward_emu(m, x, ck))))
            for name, fn in cfgs:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                base_mem = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                fn()
                torch.cuda.synchronize()
                mem = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20   # working memory of one inference
                med, p90 = time_gpu(fn, reps)
                gmed, gp90 = time_gpu(fn, reps, graph=True)
                res[f"B{B}/{name}"] = dict(ms=med, p90=p90, img_per_s=1000.0 * B / med, mem_MB=mem,
                                           graph_ms=gmed, graph_img_per_s=(1000.0 * B / gmed) if gmed else None)
                gtxt = f"   graph {gmed:8.3f} ms ({1000.0 * B / gmed:8.1f} img/s)" if gmed else ""
                print(f"  B={B:3d} {name:16s} eager {med:8.3f} ms ({1000.0 * B / med:8.1f} img/s){gtxt}"
                      f"  work mem {mem:7.1f} MB", flush=True)
        # sanity: emulated model on the GPU against the NumPy emulator, and false alarms
        ck = Checker("twosided")
        z = forward_emu(m, X, ck).float().cpu().numpy()
        res["false_alarms_twosided"] = int(ck.alarm.sum().item())
        res["n_sanity"] = int(X.shape[0])
        out["A_top1_gpu"] = z.argmax(1).tolist()
    out["A"] = res
    out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__


# ----------------------------------------------------------------------
# B: certificate cost
# ----------------------------------------------------------------------
def part_B(a, w, imgs, cal, out):
    import dcft.cert_rel as CR
    gs = None if cal["gsize"] < 0 else cal["gsize"]
    sites = [s for s, en in zip(cal["sites"], cal["enabled"]) if any(en) and w.sites[s][0] >= w.depth - 1]
    names = [f"{w.sites[s][0]}.{w.sites[s][1]}" for s in sites]
    print("  enabled sites:", names)
    if not sites:
        out["B"] = dict(note="no enabled site")
        return
    first = min([w.sites[s][0] for s in sites if w.sites[s][1] != "head"] or [w.depth - 1])
    backs = ["numpy"]
    try:
        import cupy  # noqa: F401
        backs.append("cupy")
    except Exception as e:
        print("  CuPy not available, GPU certificate skipped:", e)
    res = {b: {n: [] for n in names} for b in backs}
    res_build = {b: [] for b in backs}
    agree = {n: [] for n in names}
    for k, img in enumerate(imgs[:a.n_cert]):
        run = hw_forward_g(w, img, gs)
        ths = {s: theta_rows(gemm_of(run, w.sites[s], w.depth)) for s in sites}
        passes = {}
        for bk in backs:
            CR.set_backend(bk)
            sync = (lambda: __import__("cupy").cuda.Stream.null.synchronize()) if bk == "cupy" else (lambda: None)
            if bk == "cupy" and k == 0:          # warm-up (kernel compilation)
                rc = CR.RelCert(w, run, first_block=first)
                for s in sites:
                    rc.certify(s, ths[s])
                sync()
            t0 = time.perf_counter()
            rc = CR.RelCert(w, run, first_block=first)
            sync()
            res_build[bk].append(time.perf_counter() - t0)
            for s, n in zip(sites, names):
                t1 = time.perf_counter()
                ok = bool(rc.certify(s, ths[s]))
                sync()
                res[bk][n].append(time.perf_counter() - t1)
                passes[(bk, n)] = ok
        for n in names:
            if "cupy" in backs:
                agree[n].append(passes[("numpy", n)] == passes[("cupy", n)])
        print(f"  certificate image {k + 1}/{a.n_cert}", flush=True)
    CR.set_backend("numpy")
    out["B"] = dict(sites=names, backends=backs,
                    build_ms={b: 1e3 * float(np.median(v)) for b, v in res_build.items()},
                    cert_ms={b: {n: 1e3 * float(np.median(v)) for n, v in res[b].items()} for b in backs},
                    agree={n: (float(np.mean(v)) if v else None) for n, v in agree.items()})
    for b in backs:
        tot = out["B"]["build_ms"][b] + sum(out["B"]["cert_ms"][b].values())
        print(f"  {b:6s}: build {out['B']['build_ms'][b]:.1f} ms, per site "
              + ", ".join(f"{n} {v:.1f} ms" for n, v in out["B"]["cert_ms"][b].items())
              + f"; total {tot:.1f} ms per inference")
        out["B"][f"total_ms_{b}"] = tot


# ----------------------------------------------------------------------
# C: expected cost under a fault rate
# ----------------------------------------------------------------------
def part_C(a, out, w, cal):
    """Expected time per inference at fault rate lambda (faults per inference):
         T = T_checks + T_cert_always + lambda * (p_cert_on_alarm * T_cert_site
                                                  + p_recompute * T_recompute)
    with T_recompute either a full inference or one GEMM (T_inference / 49)."""
    if not a.e3b or not os.path.exists(a.e3b):
        print("  no E3b file, part C skipped")
        return
    R = list(csv.DictReader(open(a.e3b)))
    pol = ("P0", "P0g", "P1", "P3", "P3f")
    pol = tuple(p for p in pol if f"out_{p}" in R[0])
    prec = {p: float(np.mean([r[f"out_{p}"] == "recompute" for r in R])) for p in pol}
    psw = {p: float(np.mean([r[f"out_{p}"] == "silent_wrong" for r in R])) for p in pol}
    cand = set(cal["sites"])
    p_alarm_cand = float(np.mean([(int(r["site"]) in cand) and r["alarm_P0g"] == "True" for r in R]))
    A = out.get("A", {})
    Bc = out.get("B", {})
    bk = "cupy" if "total_ms_cupy" in Bc else "numpy"
    t_cert_all = Bc.get(f"total_ms_{bk}", 0.0)
    per_site = Bc.get("cert_ms", {}).get(bk, {})
    t_cert_site = float(np.mean(list(per_site.values()))) + Bc.get("build_ms", {}).get(bk, 0.0) if per_site else 0.0
    tm = lambda k: (A.get(k, {}).get("graph_ms") or A.get(k, {}).get("ms"))
    t_inf = tm("B1/emu+none")
    cfg = {"P0": "B1/emu+col"}
    lam = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    table = {}
    for p in pol:
        t_chk = tm(cfg.get(p, "B1/emu+grouped"))
        always = t_cert_all if p in ("P1", "P3") else 0.0
        on_alarm = p_alarm_cand * t_cert_site if p == "P3f" else 0.0
        row = dict(p_recompute=prec[p], p_silent_wrong=psw[p], cert_ms_always=always,
                   cert_ms_per_fault=on_alarm)
        if t_inf is not None and t_chk is not None:
            for gran, trec in (("inference", t_inf), ("gemm", t_inf / 49.0)):
                row[f"ms_{gran}"] = {str(l): t_chk + always + l * (on_alarm + prec[p] * trec) for l in lam}
        table[p] = row
    out["C"] = dict(n_injections=len(R), lambdas=lam, backend=bk, table=table,
                    p_alarm_candidate=p_alarm_cand)
    print(f"  {len(R)} injections; certificate backend {bk}; "
          f"P(alarm at a candidate site | fault) = {100 * p_alarm_cand:.1f}%")
    for p in pol:
        r = table[p]
        print(f"  {p:4s} recompute {100 * r['p_recompute']:5.1f}%  silent-wrong {100 * r['p_silent_wrong']:5.2f}%"
              f"  certificate {r['cert_ms_always']:.2f} ms/inference + {r['cert_ms_per_fault']:.2f} ms/fault")
        for gran in ("inference", "gemm"):
            if f"ms_{gran}" in r:
                print(f"       recompute = {gran:9s}: ms/inference at lambda = "
                      + ", ".join(f"{l:g}: {v:.3f}" for l, v in zip(lam, r[f'ms_{gran}'].values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--calib", default="results_v5/calib_v5.json")
    ap.add_argument("--e3b", default="results_v5/e3b_injections.csv")
    ap.add_argument("--offset", type=int, default=200)
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--n-cert", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--skip", default="", help="comma list of parts to skip, e.g. A")
    ap.add_argument("--out", default="results_v5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = Weights(a.weights)
    imgs = [img for img, _ in image_source(a.images, a.synthetic, a.offset, a.n_images)]
    out = dict(platform=platform.platform(), cpu=platform.processor(), numpy=np.__version__)
    skip = set(a.skip.split(",")) if a.skip else set()
    if "A" not in skip:
        print("Part A: inference and checks")
        try:
            part_A(a, imgs, out)
            # top-1 agreement of the GPU model with the NumPy emulator
            top_np = [int(np.argmax(hw_forward_g(w, im)["head"]["z"])) for im in imgs]
            out["A"]["top1_agree_numpy"] = float(np.mean(np.array(top_np) == np.array(out["A_top1_gpu"])))
            print(f"  GPU emulated model agrees with the NumPy emulator on top-1 for "
                  f"{100 * out['A']['top1_agree_numpy']:.1f}% of {len(imgs)} images; "
                  f"two-sided false alarms on them: {out['A']['false_alarms_twosided']}")
        except Exception as e:
            print("  part A failed:", repr(e))
            out["A_error"] = repr(e)
    if "B" not in skip:
        print("Part B: certificate")
        cal = json.load(open(a.calib))
        try:
            part_B(a, w, imgs, cal, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  part B failed:", repr(e))
            out["B_error"] = repr(e)
    if "C" not in skip:
        print("Part C: expected cost under a fault rate")
        part_C(a, out, w, json.load(open(a.calib)))
    json.dump(out, open(os.path.join(a.out, "bench_gpu.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "bench_gpu.json"))


if __name__ == "__main__":
    main()
