"""E3b: single bit flips in checked accumulators (the fault contract), with
grouped checks and the relational certificate.

Policies (decisions are compared with the fault-free run):
  P0   the paper's baseline: one column checksum over all rows, sound
       rounding-only threshold tau = (1+u) eta; detection -> recompute.
  P0g  the same thresholds on grouped checks (class token on its own).
  P1   grouped checks with certified-window thresholds at the (site, class)
       pairs enabled by E2b, rounding-only elsewhere; an undetected fault at
       an enabled site is certified with the a posteriori bound, failure ->
       recompute.
  P3   residual certificate. At enabled sites the certificate is run on
       every inference with the a posteriori bound of the observed residual,
       whatever the residual; at the other candidate sites it is run only
       when the rounding-only check raises an alarm. A passing certificate
       keeps the decision, a failing one triggers recomputation; with no
       alarm at a non-enabled site the behaviour is that of P0g. P3 therefore
       never recomputes more often than P0g.
  P3f  alarm-only mode of P3: the certificate runs only when the
       rounding-only check raises an alarm at a candidate site, to decide
       whether the recomputation is needed. No cost without an alarm.

Relaxed empirical thresholds (the principle of ApproxABFT/AdaptABFT) are
evaluated afterwards from the recorded residual ratios (analyse_v5.py).

Recorded for every injection: fault size, the bounds of every policy,
bounded-or-detected checks, certificate results and outcomes.
--focus restricts injections to the candidate sites, to grow the number of
certified decisions; --offset/--n-images choose the test images.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from dcft.arith import U32
from dcft.model import Weights
from dcft.faultsim import hw_forward_g, hw_forward_faulty_g, gemm_of, theta_rows
from dcft.cert_rel import RelCert
from dcft.policy import (CLASSES, thresholds_P1, theta_from_tau, alarm, ungrouped, fault_group)
from common import image_source


def outcome(alarm_, cert, changed):
    if alarm_:
        return "recompute"
    if cert is True:
        return "CERT_WRONG" if changed else "certified"
    if cert is False:
        return "recompute"
    return "silent_wrong" if changed else "silent_ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--calib", default="results_v5/calib_v5.json")
    ap.add_argument("--offset", type=int, default=200)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--inj-per-image", type=int, default=30)
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--row0", type=float, default=0.0,
                    help="probability of injecting into the class-token row (stratified sampling; 0 = uniform rows)")
    ap.add_argument("--audit-every", type=int, default=25)
    ap.add_argument("--cand", default="last1", choices=["last1", "calib"],
                    help="sites where the certificate is used: the last block and the head, or all calibrated sites")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results_v5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = Weights(a.weights)
    cal = json.load(open(a.calib))
    gs = None if cal["gsize"] < 0 else cal["gsize"]
    cand = list(cal["sites"])
    if a.cand == "last1":
        cand = [x for x in cand if w.sites[x][0] >= w.depth - 1]
    theta_cal = {s: {c: (np.inf if (v is not None and v >= 1e299) else v)
                     for c, v in zip(CLASSES, row)} for s, row in zip(cal["sites"], cal["theta"])}
    enabled = {s: {c: e for c, e in zip(CLASSES, row)} for s, row in zip(cal["sites"], cal["enabled"])}
    first_cand = min(w.sites[s][0] for s in cand if w.sites[s][1] != "head")
    rng = np.random.default_rng(a.seed)
    pool_sites = cand if a.focus else list(range(len(w.sites)))
    rows, audit_log, n_cert = [], [], 0
    name = f"e3b_injections{a.tag}.csv"
    out_csv = os.path.join(a.out, name)
    t_start = time.time()
    for k, (img, lab) in enumerate(image_source(a.images, a.synthetic, a.offset, a.n_images)):
        clean = hw_forward_g(w, img, gs)
        z0 = clean["head"]["z"]
        dec_gold = int(np.argmax(z0))
        zs = np.sort(z0.astype(np.float64))
        margin = float(zs[-1] - zs[-2])
        for _ in range(a.inj_per_image):
            si = int(rng.choice(pool_sites))
            site = w.sites[si]
            lin = w.lin(site)
            g0 = gemm_of(clean, site, w.depth)
            m = g0["acc"].shape[0]
            i = 0 if (m > 1 and rng.random() < a.row0) else int(rng.integers(m))
            j, bit = int(rng.integers(lin.N)), int(rng.integers(32))
            f = dict(kind="acc", block=site[0], stage=site[1], row=i, col=j, bit=bit)
            fr = hw_forward_faulty_g(w, clean, [f], gs)
            g = gemm_of(fr, site, w.depth)
            orig, new = float(g0["acc"][i, j]), float(g["acc"][i, j])
            delta = abs(new - orig) if np.isfinite(new) else float("inf")
            zf = fr["head"]["z"]
            dec_f = int(np.argmax(zf)) if np.all(np.isfinite(zf)) else -1
            changed = dec_f != dec_gold
            grp = fault_group(g, i)
            res = dict(image=k + a.offset, site=si, block=site[0], stage=site[1], row=i, col=j, bit=bit,
                       group=grp, delta=delta, eta_col=float(g["eta"][grp, j]), margin=margin,
                       dec_gold=dec_gold, dec_faulty=dec_f, changed=changed,
                       masked=bool(np.array_equal(zf, z0)),
                       structural=bool(site[0] == w.depth - 1 and site[1] in ("proj", "fc1", "fc2") and i != 0))
            # P0: ungrouped baseline
            ug = ungrouped(g, lin)
            res["eta_ungrouped"] = float(ug["eta"][j])
            res["alarm_P0"] = ug["alarm"]
            res["bod_P0"] = bool(ug["alarm"] or delta <= ug["theta"][j])
            # P0g
            tau0 = (1.0 + U32) * g["eta"]
            a0 = alarm(g, tau0)
            th0 = theta_from_tau(g, tau0)
            res["alarm_P0g"] = a0
            with np.errstate(invalid="ignore", divide="ignore"):
                res["rratio_g"] = float(np.nanmax(np.abs(g["r"].astype(np.float64)) / tau0)) \
                    if not g["nonfinite"] else float("inf")
                res["rratio_ug"] = float(np.nanmax(np.abs(ug["r"].astype(np.float64)) / ((1.0 + U32) * ug["eta"]))) \
                    if not g["nonfinite"] else float("inf")
            res["bod_P0g"] = bool(a0 or delta <= th0[grp, j])
            # a posteriori bound (valid for any single-entry fault, detected or not)
            thr = theta_rows(g)
            res["theta_obs"] = float(thr[i, j])
            res["bod_obs"] = bool((not np.isfinite(thr[i, j])) or delta <= thr[i, j])
            # P1
            if si in cand:
                tcls = {c: (theta_cal[si][c] if enabled[si][c] else None) for c in CLASSES}
            else:
                tcls = {}
            tau1 = thresholds_P1(g, tcls) if tcls else tau0
            a1 = alarm(g, tau1)
            th1 = theta_from_tau(g, tau1)
            res["alarm_P1"] = a1
            res["bod_P1"] = bool(a1 or delta <= th1[grp, j])
            any_en = si in cand and any(enabled[si].values())
            need_cert_P1 = (not a1) and any_en
            finite = bool(np.all(np.isfinite(thr)) and not g["nonfinite"])
            need_cert_P3 = si in cand and finite and (any_en or a0)
            cert = None
            t_c = np.nan
            if need_cert_P1 or need_cert_P3:
                aud = None
                if a.audit_every and n_cert % a.audit_every == 0:
                    aud = dict(golden=clean, run=fr, t=i, j=j, log=audit_log)
                t1 = time.perf_counter()
                rc = RelCert(w, fr, first_block=min(first_cand, site[0]) if site[1] != "head" else first_cand,
                             audit=aud)
                cert = bool(rc.certify(si, thr, "entry"))
                t_c = time.perf_counter() - t1
                n_cert += 1
            res["cert"] = "" if cert is None else cert
            res["t_cert"] = t_c
            res["out_P0"] = outcome(res["alarm_P0"], None, changed)
            res["out_P0g"] = outcome(a0, None, changed)
            res["out_P1"] = outcome(a1, cert if need_cert_P1 else None, changed)
            if si in cand and need_cert_P3:
                res["out_P3"] = outcome(False, cert, changed)
            elif si in cand and not finite:
                res["out_P3"] = "recompute"
            else:
                res["out_P3"] = res["out_P0g"]
            # P3f: certificate only when the rounding-only check raises an alarm
            if si in cand and a0 and finite:
                res["out_P3f"] = outcome(False, cert, changed)
            else:
                res["out_P3f"] = res["out_P0g"]
            rows.append(res)
        el = time.time() - t_start
        print(f"E3b image {k + 1}/{a.n_images}: {len(rows)} injections, {el / 60:.1f} min", flush=True)
        if (k + 1) % 5 == 0 or k + 1 == a.n_images:
            with open(out_csv, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    viol = sum(v for _, v, _ in audit_log)
    checks = sum(n for _, _, n in audit_log)
    np.savez(os.path.join(a.out, f"e3b_audit{a.tag}.npz"), audit=np.array([viol, checks]))
    bad = sum(r[f"out_{p}"] == "CERT_WRONG" for r in rows for p in ("P1", "P3", "P3f"))
    bod = sum(not (r["bod_P0"] and r["bod_P0g"] and r["bod_P1"] and r["bod_obs"]) for r in rows)
    print(f"E3b finished: {len(rows)} injections, certified-wrong {bad}, "
          f"bounded-or-detected violations {bod}, audit violations {viol}/{checks}")
    for p in ("P0", "P0g", "P1", "P3", "P3f"):
        oc = {}
        for r in rows:
            oc[r[f"out_{p}"]] = oc.get(r[f"out_{p}"], 0) + 1
        print(f"  {p:4s}", dict(sorted(oc.items())))


if __name__ == "__main__":
    main()
