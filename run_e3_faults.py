"""E3: single bit flips in checked GEMM accumulators.

One fault per inference: a uniformly chosen checked GEMM (48 in the blocks
plus the classifier head), output entry and bit of the binary32 accumulator.
Decisions are compared with the fault-free (golden) run of the same
emulated accelerator.

Policies
  P0  sound rounding-only thresholds, tau = (1+u) eta (no certificate);
  P1  certified-window thresholds where the site has a calibrated budget,
      otherwise P0, plus the per-site decision certificate when undetected,
      using the a posteriori fault bound from the observed residual;
  P2  the P1 thresholds without the certificate.

Recorded for every injection: the fault magnitude, whether each policy
detects it, whether the bounded-or-detected property held (an undetected
fault never exceeds the proved bound), the certificate result, and the
decision. Every `audit_every`-th certified case is audited stage by stage
against the golden run.
"""
import argparse
import csv
import json
import os

import numpy as np

from dcft.model import Weights, hw_forward, hw_forward_faulty, eta_bound, thresholds, site_gemm, theta_observed
from dcft.certificate import Stats, single_site_pass
from common import image_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--calib", default="results/calib.json")
    ap.add_argument("--offset", type=int, default=None)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--inj-per-image", type=int, default=30)
    ap.add_argument("--audit-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    w = Weights(a.weights)
    calib = json.load(open(a.calib))
    theta_sites = np.array(calib["theta_sites"])
    offset = calib["n_cal"] if a.offset is None else a.offset
    rng = np.random.default_rng(a.seed)
    S = len(w.sites)
    rows, audit_log, n_cert = [], [], 0
    out_csv = os.path.join(a.out, "e3_injections.csv")
    for k, (img, lab) in enumerate(image_source(a.images, a.synthetic, offset, a.n_images)):
        clean = hw_forward(w, img)
        z_clean = clean["head"]["z"]
        dec_gold = int(np.argmax(z_clean))
        zs = np.sort(z_clean.astype(np.float64))
        margin_gold = float(zs[-1] - zs[-2])
        for _ in range(a.inj_per_image):
            si = int(rng.integers(S))
            site = w.sites[si]
            lin = w.lin(site)
            g0 = site_gemm(w, clean, site)
            Tm = g0["acc"].shape[0]
            i, j, bit = int(rng.integers(Tm)), int(rng.integers(lin.N)), int(rng.integers(32))
            orig = float(g0["acc"][i, j])
            fr = hw_forward_faulty(w, clean, (site[0], site[1], i, j, bit))
            g = site_gemm(w, fr, site)
            new = float(g["acc"][i, j])
            delta = abs(new - orig) if np.isfinite(new) else float("inf")
            eta = eta_bound(g["omega_hat"], g["a16f"].shape[0], lin.K)
            m = g["a16f"].shape[0]
            r = np.abs(g["r"].astype(np.float64))
            z_f = fr["head"]["z"]
            dec_f = int(np.argmax(z_f)) if np.all(np.isfinite(z_f)) else -1
            res = dict(image=k + offset, site=si, block=site[0], stage=site[1], row=i, col=j, bit=bit,
                       orig=orig, delta=delta, eta_col=float(eta[j]), eta_max=float(eta.max()),
                       theta_site=float(theta_sites[si]), margin_gold=margin_gold,
                       dec_gold=dec_gold, dec_faulty=dec_f,
                       changed=bool(dec_f != dec_gold), masked=bool(np.array_equal(z_f, z_clean)))
            for pol in ("P0", "P1"):
                ts = None if (pol == "P0" or theta_sites[si] <= 0) else theta_sites[si]
                tau, th_act = thresholds(eta, ts, "P0" if ts is None else "P1", m)
                alarm = bool(g["nonfinite"] or np.any(r > tau))
                res[f"alarm_{pol}"] = alarm
                res[f"theta_{pol}"] = th_act
                res[f"bod_{pol}"] = bool(alarm or delta <= th_act)     # bounded-or-detected held
            cert = ""
            if not res["alarm_P1"] and theta_sites[si] > 0:
                st = Stats(w, fr)
                aud = None
                if a.audit_every and (n_cert % a.audit_every == 0):
                    aud = dict(golden=clean, run=fr, row=0, log=audit_log)
                th_obs = theta_observed(g["r"], eta, m)       # <= theta_P1 when undetected
                res["theta_obs"] = th_obs
                cert = single_site_pass(st, si, th_obs, aud)
                n_cert += 1
            res.setdefault("theta_obs", "")
            res["cert_P1"] = cert
            ch = res["changed"]
            res["out_P0"] = "detected" if res["alarm_P0"] else ("silent_wrong" if ch else "silent_ok")
            res["out_P2"] = "detected" if res["alarm_P1"] else ("silent_wrong" if ch else "silent_ok")
            if res["alarm_P1"]:
                res["out_P1"] = "detected"
            elif theta_sites[si] <= 0:
                res["out_P1"] = res["out_P0"]          # no certified budget at this site
            elif cert:
                res["out_P1"] = "CERT_WRONG" if ch else "certified"
            else:
                res["out_P1"] = "fallback"
            rows.append(res)
        print(f"E3 image {k + 1}/{a.n_images}: {len(rows)} injections", flush=True)
        if (k + 1) % 10 == 0 or k + 1 == a.n_images:
            with open(out_csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    viol = sum(v for _, v, _ in audit_log)
    checks = sum(n for _, _, n in audit_log)
    np.savez(os.path.join(a.out, "e3_audit.npz"), audit=np.array([viol, checks]))
    bad = sum(r["out_P1"] == "CERT_WRONG" for r in rows)
    bod = sum(not (r["bod_P0"] and r["bod_P1"]) for r in rows)
    print(f"E3 finished: {len(rows)} injections, certified-wrong {bad}, "
          f"bounded-or-detected violations {bod}, audit violations {viol}/{checks}")


if __name__ == "__main__":
    main()
