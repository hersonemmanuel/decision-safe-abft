"""E2b: budgets and windows with the relational certificate and grouped checks.

Calibration split: for every image, candidate site and row class
(class token / other rows), the largest uniform fault bound theta* that the
relational certificate tolerates, found by bisection in log space. The
window of a (site, class) is open when
    (1 - Gamma) theta* >= (1 + (1+u)/(1-u)) max eta
over that class's groups. (site, class) pairs whose window is open on at
least a 1-q share of the calibration images are enabled for policy P1.

Test split, fault-free: false alarms of the grouped rounding-only checks
(theory: none), the in-situ ratio |r|/eta, and, for policy P3, how often the
runtime certificate with the observed a posteriori bound passes on a
fault-free run (a failure there would cost a recomputation) and how long it
takes.

Use --workers to spread images over CPU cores.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from dcft.arith import U32
from dcft.model import Weights, ref_forward
from dcft.faultsim import hw_forward_g, gemm_of, theta_rows
from dcft.cert_rel import RelCert
from dcft.policy import CLASSES, uniform_theta, window_ratio, alarm
from common import image_source

G = {}


def candidate_sites(w, which):
    L = w.depth
    if which == "all":
        return list(range(len(w.sites)))
    blocks = {"last1": [L - 1], "last2": [L - 2, L - 1], "last3": [L - 3, L - 2, L - 1]}[which]
    return [i for i, (b, s) in enumerate(w.sites) if b in blocks] + [len(w.sites) - 1]


def structural_harmless(w, site, cls):
    b, s = w.sites[site]
    return cls == "rest" and b == w.depth - 1 and s in ("proj", "fc1", "fc2")


def theta_star(rc, site, gm, cls, lo=-10.0, hi=2.0, steps=9):
    if len(gm["groups"]) == 1 and cls == "rest":
        return np.nan
    f = lambda e: rc.certify(site, uniform_theta(gm, cls, 10.0 ** e), "entry")
    if not f(lo):
        return 0.0
    if f(hi):
        return 10.0 ** hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid):
            lo = mid
        else:
            hi = mid
    return 10.0 ** lo


def _init(weights, gsize, sites):
    G["w"] = Weights(weights)
    G["gsize"] = gsize
    G["sites"] = sites


def _cal_one(args):
    k, img = args
    w, gs, sites = G["w"], G["gsize"], G["sites"]
    run = hw_forward_g(w, img, gs)
    first = min(w.sites[s][0] for s in sites if w.sites[s][1] != "head") if any(
        w.sites[s][1] != "head" for s in sites) else w.depth - 1
    t0 = time.perf_counter()
    rc = RelCert(w, run, first_block=first)
    th = np.full((len(sites), 2), np.nan)
    wr = np.full((len(sites), 2), np.nan)
    for i, si in enumerate(sites):
        gm = gemm_of(run, w.sites[si], w.depth)
        for c, cls in enumerate(CLASSES):
            if structural_harmless(w, si, cls):
                th[i, c], wr[i, c] = np.inf, np.inf
                continue
            ts = theta_star(rc, si, gm, cls)
            th[i, c] = ts
            wr[i, c] = window_ratio(gm, cls, ts) if np.isfinite(ts) else np.nan
    return k, th, wr, time.perf_counter() - t0


def _test_one(args):
    k, img, lab = args
    w, gs, sites = G["w"], G["gsize"], G["sites"]
    t0 = time.perf_counter()
    run = hw_forward_g(w, img, gs)
    t_fwd = time.perf_counter() - t0
    ref = ref_forward(w, img)
    fa, ratios = 0, np.zeros(len(w.sites))
    for i, site in enumerate(w.sites):
        gm = gemm_of(run, site, w.depth)
        fa += int(alarm(gm, (1.0 + U32) * gm["eta"]))
        ratios[i] = float(np.max(np.abs(gm["r"].astype(np.float64)) / gm["eta"]))
    first = min(w.sites[s][0] for s in sites if w.sites[s][1] != "head")
    rc = RelCert(w, run, first_block=first)
    cert = np.zeros(len(sites), bool)
    tc = np.zeros(len(sites))
    for i, si in enumerate(sites):
        gm = gemm_of(run, w.sites[si], w.depth)
        t1 = time.perf_counter()
        cert[i] = rc.certify(si, theta_rows(gm), "entry")
        tc[i] = time.perf_counter() - t1
    dh, dr = int(np.argmax(run["head"]["z"])), int(np.argmax(ref["z"]))
    zs = np.sort(run["head"]["z"].astype(np.float64))
    return dict(k=k, fa=fa, ratios=ratios, cert=cert, tc=tc, agree=dh == dr,
                top1=(dh == lab, dr == lab) if lab >= 0 else None, t_fwd=t_fwd,
                margin=float(zs[-1] - zs[-2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--test-offset", type=int, default=200,
                    help="first test image (keep it equal to the paper's calibration size)")
    ap.add_argument("--gsize", type=int, default=-1, help="-1: one group for all non-class rows")
    ap.add_argument("--sites", default="last2", choices=["last1", "last2", "last3", "all"])
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results_v5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gs = None if a.gsize < 0 else a.gsize
    w = Weights(a.weights)
    sites = candidate_sites(w, a.sites)
    names = [f"{w.sites[s][0]}.{w.sites[s][1]}" for s in sites]
    cal = [(k, img) for k, (img, _) in enumerate(image_source(a.images, a.synthetic, 0, a.n_cal))]
    with Pool(a.workers, _init, (a.weights, gs, sites)) as pool:
        res = []
        for r in pool.imap_unordered(_cal_one, cal):
            res.append(r)
            print(f"E2b calibration {len(res)}/{a.n_cal} ({r[3]:.0f} s)", flush=True)
        res.sort(key=lambda r: r[0])
        TH = np.array([r[1] for r in res])
        WR = np.array([r[2] for r in res])
        tcal = np.array([r[3] for r in res])
        theta_s = np.full((len(sites), 2), np.nan)
        enabled = np.zeros((len(sites), 2), bool)
        for i in range(len(sites)):
            for c in range(2):
                col = TH[:, i, c]
                if np.all(np.isnan(col)):
                    continue
                theta_s[i, c] = np.inf if np.all(np.isinf(col)) else float(np.nanquantile(col, a.q))
                wrc = WR[:, i, c]
                enabled[i, c] = bool(np.mean(np.nan_to_num(wrc, nan=0.0) >= 1.0) >= 1.0 - a.q)
        calib = dict(q=a.q, n_cal=a.n_cal, gsize=a.gsize, sites=sites, names=names,
                     classes=list(CLASSES),
                     theta=[[None if np.isnan(v) else (1e300 if np.isinf(v) else float(v)) for v in row]
                            for row in theta_s],
                     enabled=enabled.tolist())
        with open(os.path.join(a.out, "calib_v5.json"), "w") as f:
            json.dump(calib, f, indent=1)
        print("enabled (site, class):", [(names[i], CLASSES[c]) for i in range(len(sites))
                                         for c in range(2) if enabled[i, c]])
        tests = [(k, img, lab) for k, (img, lab) in
                 enumerate(image_source(a.images, a.synthetic, a.test_offset, a.n_test))]
        out = []
        for r in pool.imap_unordered(_test_one, tests):
            out.append(r)
            print(f"E2b test {len(out)}/{a.n_test}", flush=True)
    out.sort(key=lambda r: r["k"])
    fa = sum(r["fa"] for r in out)
    ratios = np.max([r["ratios"] for r in out], axis=0)
    cert = np.array([r["cert"] for r in out])
    tc = np.array([r["tc"] for r in out])
    top1 = np.array([r["top1"] for r in out if r["top1"] is not None]) if any(
        r["top1"] is not None for r in out) else np.zeros((0, 2))
    np.savez(os.path.join(a.out, "e2b.npz"), theta=TH, window_ratio=WR, theta_s=theta_s,
             enabled=enabled, t_cal=tcal, sites=np.array(sites), names=np.array(names),
             false_alarms=fa, n_checks=a.n_test * len(w.sites), ratio_max=ratios,
             cert_clean=cert, t_cert=tc, agree=np.array([r["agree"] for r in out]),
             top1=top1, t_fwd=np.array([r["t_fwd"] for r in out]),
             margin=np.array([r["margin"] for r in out]), depth=w.depth, gsize=a.gsize)
    print(f"E2b finished. false alarms {fa}/{a.n_test * len(w.sites)}; max |r|/eta {ratios.max():.3g}")
    for i, n in enumerate(names):
        print(f"  {n:9s} cls theta_s {theta_s[i, 0]:.3g} rest {theta_s[i, 1]:.3g} "
              f"enabled {enabled[i].tolist()}  clean-run certified {cert[:, i].mean() * 100:.1f}%  "
              f"t_cert {np.median(tc[:, i]) * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
