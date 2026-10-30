"""E4: fault models beyond single accumulator bit flips.

Kinds (status with respect to the theory):
  acc      single bit flip in an accumulator entry           inside the contract
  oprow    bit flip of a binary16 operand a[i, k] used by     inside (row class)
           the main product only: perturbs row i
  wcol     bit flip of a binary16 weight W[k, j] used by the  inside for two-sided
           main product only: perturbs column j               checks; outside for
                                                              column checks alone
  shared   bit flip of an operand seen by both the product    outside (shared operand)
           and its checksum
  double   two accumulator bit flips in the same GEMM         outside (single event)
  twosite  two accumulator bit flips at two different GEMMs   outside (single event)

For every injection: detection by the grouped column checks and by
two-sided checks (column + row), whether the bounded-or-detected property
held (each faulty entry within the a posteriori bound of a check that sees it
alone), whether the decision changed, and, for the kinds the certificate
covers (acc: entry class; oprow: row class) at candidate sites, the
certificate result with its audit.
"""
import argparse
import csv
import json
import os

import numpy as np

from dcft.arith import U32
from dcft.model import Weights
from dcft.faultsim import (hw_forward_g, hw_forward_faulty_g, gemm_of, theta_rows, alarm_P0, alarm_row)
from dcft.cert_rel import RelCert
from common import image_source

KINDS = ("acc", "oprow", "wcol", "shared", "double", "twosite")
INSIDE = {"acc": "inside", "oprow": "inside", "wcol": "inside (two-sided only)",
          "shared": "outside", "double": "outside", "twosite": "outside"}


def make_fault(rng, w, clean, si, kind):
    site = w.sites[si]
    lin = w.lin(site)
    m = gemm_of(clean, site, w.depth)["acc"].shape[0]
    base = dict(block=site[0], stage=site[1])
    if kind == "acc":
        return [dict(base, kind="acc", row=int(rng.integers(m)), col=int(rng.integers(lin.N)),
                     bit=int(rng.integers(32)))]
    if kind == "oprow":
        return [dict(base, kind="oprow", row=int(rng.integers(m)), k=int(rng.integers(lin.K)),
                     bit=int(rng.integers(16)))]
    if kind == "wcol":
        return [dict(base, kind="wcol", k=int(rng.integers(lin.K)), col=int(rng.integers(lin.N)),
                     bit=int(rng.integers(16)))]
    if kind == "shared":
        return [dict(base, kind="shared", row=int(rng.integers(m)), k=int(rng.integers(lin.K)),
                     bit=int(rng.integers(16)))]
    if kind == "double":
        return [dict(base, kind="acc", row=int(rng.integers(m)), col=int(rng.integers(lin.N)),
                     bit=int(rng.integers(32))) for _ in range(2)]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--calib", default="results_v5/calib_v5.json")
    ap.add_argument("--offset", type=int, default=500)
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--inj-per-image", type=int, default=30)
    ap.add_argument("--audit-every", type=int, default=10)
    ap.add_argument("--cand", default="last1", choices=["last1", "calib"],
                    help="sites where the certificate is used: the last block and the head, or all calibrated sites")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="results_v5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = Weights(a.weights)
    cal = json.load(open(a.calib))
    gs = None if cal["gsize"] < 0 else cal["gsize"]
    cand = set(cal["sites"])
    if a.cand == "last1":
        cand = {x for x in cand if w.sites[x][0] >= w.depth - 1}
    first_cand = min(w.sites[s][0] for s in cand if w.sites[s][1] != "head")
    rng = np.random.default_rng(a.seed)
    S = len(w.sites)
    rows, audit_log, n_aud = [], [], 0
    for k, (img, _) in enumerate(image_source(a.images, a.synthetic, a.offset, a.n_images)):
        clean = hw_forward_g(w, img, gs, two_sided=True)
        dec_gold = int(np.argmax(clean["head"]["z"]))
        for n in range(a.inj_per_image):
            kind = KINDS[n % len(KINDS)]
            if kind == "twosite":
                s1, s2 = rng.choice(S, 2, replace=False)
                faults = make_fault(rng, w, clean, int(s1), "acc") + make_fault(rng, w, clean, int(s2), "acc")
                sites = [int(s1), int(s2)]
            else:
                si = int(rng.integers(S))
                faults = make_fault(rng, w, clean, si, kind)
                sites = [si]
            fr = hw_forward_faulty_g(w, clean, faults, gs, two_sided=True)
            zf = fr["head"]["z"]
            dec_f = int(np.argmax(zf)) if np.all(np.isfinite(zf)) else -1
            det_col = det_two = False
            bod_col = bod_two = True
            maxratio_col = 0.0
            for si in sites:
                site = w.sites[si]
                g = gemm_of(fr, site, w.depth)
                with np.errstate(invalid="ignore", over="ignore"):
                    Dl = np.abs(g["delta_site"])          # this GEMM's own fault, same input
                Dl = np.where(np.isnan(Dl), np.inf, Dl)
                ac, ar = alarm_P0(g), alarm_row(g)
                det_col |= ac
                det_two |= (ac or ar)
                thc = theta_rows(g)                                # column-check bound per entry
                thr = np.broadcast_to(g["theta_row"][:, None], Dl.shape)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rc_ = np.where(Dl > 0, Dl / thc, 0.0)
                maxratio_col = max(maxratio_col, float(np.nanmax(rc_)))
                if not ac:
                    bod_col &= bool(np.all(Dl <= thc))
                if not (ac or ar):
                    bod_two &= bool(np.all((Dl <= thc) | (Dl <= thr)))
            res = dict(image=k + a.offset, kind=kind, status=INSIDE[kind],
                       sites=";".join(f"{w.sites[s][0]}.{w.sites[s][1]}" for s in sites),
                       masked=bool(np.array_equal(zf, clean["head"]["z"])), changed=dec_f != dec_gold,
                       det_col=det_col, det_two=det_two, bod_col=bod_col, bod_two=bod_two,
                       max_ratio_col=maxratio_col)
            cert = ""
            if kind in ("acc", "oprow") and sites[0] in cand:
                site = w.sites[sites[0]]
                g = gemm_of(fr, site, w.depth)
                th = theta_rows(g)
                if np.all(np.isfinite(th)) and not g["nonfinite"]:
                    f0 = faults[0]
                    aud = None
                    if a.audit_every and n_aud % a.audit_every == 0:
                        aud = dict(golden=clean, run=fr, t=f0["row"], j=f0.get("col", -1), log=audit_log)
                    n_aud += 1
                    rc = RelCert(w, fr, first_block=min(first_cand, site[0]) if site[1] != "head" else first_cand,
                                 audit=aud)
                    cert = bool(rc.certify(sites[0], th, "entry" if kind == "acc" else "row"))
            res["cert"] = cert
            res["cert_wrong"] = bool(cert is True and res["changed"])
            rows.append(res)
        print(f"E4 image {k + 1}/{a.n_images}: {len(rows)} injections", flush=True)
        if (k + 1) % 5 == 0 or k + 1 == a.n_images:
            with open(os.path.join(a.out, "e4_faultmodels.csv"), "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    viol = sum(v for _, v, _ in audit_log)
    checks = sum(n for _, _, n in audit_log)
    np.savez(os.path.join(a.out, "e4_audit.npz"), audit=np.array([viol, checks]))
    print(f"E4 finished: {len(rows)} injections, audit violations {viol}/{checks}")
    print(f"{'kind':8s} {'status':24s} {'n':>5s} {'masked':>7s} {'det col':>8s} {'det 2s':>7s} "
          f"{'BOD! col':>8s} {'BOD! 2s':>8s} {'silent-wrong col':>16s} {'2s':>4s} {'cert':>5s} {'wrong':>5s}")
    for kind in KINDS:
        R = [r for r in rows if r["kind"] == kind]
        if not R:
            continue
        n = len(R)
        pct = lambda key: 100.0 * sum(bool(r[key]) for r in R) / n
        sw_c = sum((not r["det_col"]) and r["changed"] for r in R)
        sw_2 = sum((not r["det_two"]) and r["changed"] for r in R)
        print(f"{kind:8s} {INSIDE[kind]:24s} {n:5d} {pct('masked'):6.1f}% {pct('det_col'):7.1f}% "
              f"{pct('det_two'):6.1f}% {sum(not r['bod_col'] for r in R):8d} {sum(not r['bod_two'] for r in R):8d} "
              f"{sw_c:16d} {sw_2:4d} {sum(r['cert'] is True for r in R):5d} {sum(r['cert_wrong'] for r in R):5d}")


if __name__ == "__main__":
    main()
