"""E2: fault-free behaviour and calibration of certified thresholds.

Calibration split: for every image and every checked GEMM, the largest fault
bound theta* that the certificate tolerates at that site (relative to the
golden run). Per-site thresholds use the q-quantile of theta*. The window at
a site is non-empty for all columns when
    (1 - G) theta* >= eta_max (1 + (1+u)/(1-u)).

Test split, fault-free: false alarms of the sound rounding-only thresholds
(theory: none) and the in-situ ratio |r| / eta at every checked GEMM, plus
agreement of the emulated accelerator with the float64 reference and timing.
"""
import argparse
import json
import os
import time

import numpy as np

from dcft.arith import U32
from dcft.model import Weights, hw_forward, ref_forward, eta_bound, gamma_sigma, site_gemm
from dcft.certificate import Stats, theta_star
from common import image_source


def site_etas(w, run):
    out = []
    for site in w.sites:
        g = site_gemm(w, run, site)
        out.append(eta_bound(g["omega_hat"], g["a16f"].shape[0], w.lin(site).K))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--q", type=float, default=0.05, help="theta quantile level per site")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = Weights(a.weights)
    S = len(w.sites)
    u = U32
    TH, WR, tcal = [], [], []
    for k, (img, _) in enumerate(image_source(a.images, a.synthetic, 0, a.n_cal)):
        run = hw_forward(w, img)
        t0 = time.perf_counter()
        ts = theta_star(Stats(w, run), S)
        tcal.append(time.perf_counter() - t0)
        etas = site_etas(w, run)
        WR.append(np.array([(1 - gamma_sigma(site_gemm(w, run, s)["a16f"].shape[0])) * ts[i]
                            / (etas[i].max() * (1 + (1 + u) / (1 - u)))
                            for i, s in enumerate(w.sites)]))
        TH.append(ts)
        print(f"E2 calibration {k + 1}/{a.n_cal}", flush=True)
    TH, WR = np.array(TH), np.array(WR)
    theta_sites = np.quantile(TH, a.q, axis=0)
    with open(os.path.join(a.out, "calib.json"), "w") as f:
        json.dump(dict(q=a.q, n_cal=a.n_cal, theta_sites=theta_sites.tolist(),
                       sites=[f"{b}.{s}" for b, s in w.sites]), f, indent=1)
    fa, ratio_max, agree, acc, t_fwd, t_ref = 0, np.zeros(S), [], [], [], []
    labels = []
    for k, (img, lab) in enumerate(image_source(a.images, a.synthetic, a.n_cal, a.n_test)):
        t0 = time.perf_counter()
        run = hw_forward(w, img)
        t1 = time.perf_counter()
        ref = ref_forward(w, img)
        t2 = time.perf_counter()
        etas = site_etas(w, run)
        for i, site in enumerate(w.sites):
            g = site_gemm(w, run, site)
            r = np.abs(g["r"].astype(np.float64))
            tau = (1 + u) * etas[i]
            fa += int(g["nonfinite"] or np.any(r > tau))
            ratio_max[i] = max(ratio_max[i], float(np.max(r / etas[i])))
        dh, dr = int(np.argmax(run["head"]["z"])), int(np.argmax(ref["z"]))
        agree.append(dh == dr)
        if lab >= 0:
            labels.append((dh == lab, dr == lab))
        t_fwd.append(t1 - t0)
        t_ref.append(t2 - t1)
        print(f"E2 test {k + 1}/{a.n_test}", flush=True)
    np.savez(os.path.join(a.out, "e2.npz"), theta=TH, window_ratio=WR, theta_sites=theta_sites,
             false_alarms=fa, n_checks=a.n_test * S, ratio_max=ratio_max,
             agree=np.array(agree), top1=np.array(labels) if labels else np.zeros((0, 2)),
             t_fwd=np.array(t_fwd), t_cal=np.array(tcal), depth=w.depth)
    print(f"E2 finished. false alarms {fa}/{a.n_test * S}; max |r|/eta {ratio_max.max():.3g}; "
          f"sites with theta_s > 0: {int(np.sum(theta_sites > 0))}/{S}")


if __name__ == "__main__":
    main()
