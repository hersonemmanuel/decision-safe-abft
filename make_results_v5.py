"""Turn the v5 raw results into LaTeX macros, tables and figures.

Inputs (all in --res, missing ones are skipped):
  calib_v5.json, e2b.npz             E2b
  e3b_injections*.csv, e3b_audit*.npz E3b (all tags are merged)
  e4_faultmodels.csv, e4_audit.npz   E4
  bench_gpu.json                     GPU benchmark
  ../results/e2.npz (--old-e2)       the paper's interval certificate, for comparison
Outputs: <res>/macros_v5.tex, <res>/summary_v5.txt, <res>/fig_v5_window.pdf,
         <res>/fig_v5_policies.pdf
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from dcft.policy import zero_fail_bound, clopper_pearson

POL = ("P0", "P0g", "P1", "P3", "P3f")


def fmt(v, d=1):
    return f"{v:.{d}f}"


def sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    return f"{m:.2f}{{\\times}}10^{{{e}}}"


def empirical_baseline(R, sites, budget=0.0, kappas=None):
    """Relaxed thresholds tuned by fault injection (the principle of
    ApproxABFT/AdaptABFT): per site, the largest kappa such that, on the
    tuning half of the images, no more than `budget` of the injections at that
    site change the decision while staying below kappa times the rounding-only
    threshold. Evaluated on the other half."""
    if kappas is None:
        kappas = np.logspace(0, 6, 61)
    imgs = sorted({int(r["image"]) for r in R})
    half = set(imgs[: len(imgs) // 2])
    tune = [r for r in R if int(r["image"]) in half]
    test = [r for r in R if int(r["image"]) not in half]
    kap = {}
    for s in sites:
        T = [r for r in tune if int(r["site"]) == s]
        best = 1.0
        for k in kappas:
            sw = sum((float(r["rratio_g"]) <= k) and r["changed"] == "True" for r in T)
            if T and sw / len(T) <= budget:
                best = k
            else:
                break
        kap[s] = best
    out = dict(recompute=0, silent_wrong=0, silent_ok=0, n=len(test))
    for r in test:
        k = kap.get(int(r["site"]), 1.0)
        if float(r["rratio_g"]) > k:
            out["recompute"] += 1
        elif r["changed"] == "True":
            out["silent_wrong"] += 1
        else:
            out["silent_ok"] += 1
    p0 = dict(recompute=sum(r["alarm_P0g"] == "True" for r in test),
              silent_wrong=sum(r["alarm_P0g"] != "True" and r["changed"] == "True" for r in test))
    return out, p0, kap, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v5")
    ap.add_argument("--old-e2", default="results/e2.npz")
    a = ap.parse_args()
    M, S = {}, []
    say = lambda t: (S.append(t), print(t))
    cal = json.load(open(os.path.join(a.res, "calib_v5.json"))) if os.path.exists(
        os.path.join(a.res, "calib_v5.json")) else None

    # ---------------------------------------------------------------- E2b
    p = os.path.join(a.res, "e2b.npz")
    if os.path.exists(p):
        e = np.load(p, allow_pickle=True)
        names = [str(n) for n in e["names"]]
        en = e["enabled"]
        M["VfiveFalseAlarms"] = str(int(e["false_alarms"]))
        M["VfiveNumChecks"] = f"{int(e['n_checks']):,}".replace(",", "{,}")
        M["VfiveMaxRatio"] = f"{float(e['ratio_max'].max()):.3g}"
        M["VfiveNumCal"] = str(e["theta"].shape[0])
        en_list = [f"{names[i]} ({'class token' if c == 0 else 'other rows'})"
                   for i in range(len(names)) for c in range(2) if en[i, c]]
        M["VfiveEnabledNames"] = ", ".join(en_list) if en_list else "none"
        M["VfiveNumEnabled"] = str(len(en_list))
        M["VfiveNumEnabledSites"] = str(int(np.sum(en.any(1))))
        say(f"E2b: false alarms {M['VfiveFalseAlarms']}/{M['VfiveNumChecks']}, max |r|/eta {M['VfiveMaxRatio']}")
        say("E2b: enabled (site, class): " + M["VfiveEnabledNames"])
        WR = e["window_ratio"]
        for i, n in enumerate(names):
            for c, cn in enumerate(("cls", "rest")):
                col = WR[:, i, c]
                if np.all(np.isnan(col)):
                    continue
                fin = col[np.isfinite(col)]
                med = "inf" if fin.size == 0 else f"{np.median(fin):.3g}"
                say(f"   {n:9s} {cn:4s} window open {100 * np.mean(np.nan_to_num(col, nan=0) >= 1):5.1f}% "
                    f"median ratio {med}  clean-run certified {100 * e['cert_clean'][:, i].mean():5.1f}%"
                    if c == 0 else
                    f"   {n:9s} {cn:4s} window open {100 * np.mean(np.nan_to_num(col, nan=0) >= 1):5.1f}% "
                    f"median ratio {med}")
        M["VfiveCertMs"] = fmt(1e3 * float(np.median(e["t_cert"].sum(1))), 0)
        # figure: window ratio per site and class, against the paper's certificate
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.5, 2.4))
            x = np.arange(len(names))
            for c, (lab, mk) in enumerate((("class token", "o"), ("other rows", "s"))):
                col = WR[:, :, c]
                lo = np.nanquantile(np.clip(col, 1e-12, 1e12), 0.05, axis=0)
                md = np.nanquantile(np.clip(col, 1e-12, 1e12), 0.5, axis=0)
                hi = np.nanquantile(np.clip(col, 1e-12, 1e12), 0.95, axis=0)
                ax.errorbar(x + (c - 0.5) * 0.2, np.log10(md), yerr=[np.log10(md) - np.log10(lo),
                            np.log10(hi) - np.log10(md)], fmt=mk, ms=3, lw=0.8, capsize=1.5, label=lab)
            if os.path.exists(a.old_e2):
                o = np.load(a.old_e2)
                old = o["window_ratio"]
                L = int(o["depth"])
                idx = []
                for n in names:
                    b, s = n.split(".")
                    b = int(b)
                    idx.append(4 * b + ("qkv", "proj", "fc1", "fc2").index(s) if s != "head" else 4 * L)
                om = np.nanmedian(np.clip(old[:, idx], 1e-12, 1e12), axis=0)
                ax.plot(x, np.log10(om), "k_", ms=8, label="paper's certificate")
            ax.axhline(0, ls="--", c="gray", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=60, fontsize=6)
            ax.set_ylabel(r"$\log_{10}$ window ratio", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5.5, frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(a.res, "fig_v5_window.pdf"))
            plt.close(fig)
        except Exception as ex:
            say(f"(window figure skipped: {ex})")

    # ---------------------------------------------------------------- E3b
    files = sorted(glob.glob(os.path.join(a.res, "e3b_injections*.csv")))
    if files:
        R = []
        for f in files:
            R += list(csv.DictReader(open(f)))
        n = len(R)
        M["VfiveNumInj"] = f"{n:,}".replace(",", "{,}")
        M["VfiveNumImages"] = f"{len({r['image'] for r in R}):,}".replace(",", "{,}")
        bod = sum(any(r[k] != "True" for k in ("bod_P0", "bod_P0g", "bod_P1", "bod_obs")) for r in R)
        M["VfiveBodViol"] = str(bod)
        aud = [np.load(f)["audit"] for f in glob.glob(os.path.join(a.res, "e3b_audit*.npz"))]
        if aud:
            M["VfiveAuditViol"] = str(int(sum(x[0] for x in aud)))
            M["VfiveAuditChecks"] = f"{int(sum(x[1] for x in aud)):,}".replace(",", "{,}")
        say(f"E3b: {n} injections from {M['VfiveNumImages']} images; BOD violations {bod}; "
            f"audit {M.get('VfiveAuditViol', '?')}/{M.get('VfiveAuditChecks', '?')}")
        cand = set(cal["sites"]) if cal else set()
        en_sites = {s for s, e_ in zip(cal["sites"], cal["enabled"]) if any(e_)} if cal else set()
        cnt = {p: {} for p in POL if f"out_{p}" in R[0]}
        for r in R:
            for p in cnt:
                cnt[p][r[f"out_{p}"]] = cnt[p].get(r[f"out_{p}"], 0) + 1
        for p in cnt:
            say(f"   {p:4s} " + ", ".join(f"{k} {v}" for k, v in sorted(cnt[p].items())))
            M[f"Vfive{p}Recompute"] = str(cnt[p].get("recompute", 0))
            M[f"Vfive{p}SilentWrong"] = str(cnt[p].get("silent_wrong", 0))
            M[f"Vfive{p}Certified"] = str(cnt[p].get("certified", 0))
            M[f"Vfive{p}CertWrong"] = str(cnt[p].get("CERT_WRONG", 0))
        # at the enabled sites
        RE = [r for r in R if int(r["site"]) in en_sites]
        if RE:
            M["VfiveEnInj"] = f"{len(RE):,}".replace(",", "{,}")
            for p in cnt:
                M[f"Vfive{p}RecEn"] = str(sum(r[f"out_{p}"] == "recompute" for r in RE))
            r0 = sum(r["out_P0"] == "recompute" for r in RE)
            for p in ("P1", "P3", "P3f"):
                if p in cnt and r0:
                    rp = sum(r[f"out_{p}"] == "recompute" for r in RE)
                    M[f"Vfive{p}RedEn"] = fmt(100 * (r0 - rp) / r0)
            say(f"   enabled sites: {len(RE)} injections; recompute P0 {r0}, " + ", ".join(
                f"{p} {M[f'Vfive{p}RecEn']}" for p in cnt))
        # certified decisions: bounds by injection and by image
        cert = [r for r in R if r.get("out_P3") in ("certified", "CERT_WRONG") and r.get("structural") != "True"]
        nstruct = sum(r.get("out_P3") == "certified" and r.get("structural") == "True" for r in R)
        M["VfiveNumStructural"] = f"{nstruct:,}".replace(",", "{,}")
        nc = len(cert)
        ni = len({r["image"] for r in cert})
        M["VfiveNumCertified"] = f"{nc:,}".replace(",", "{,}")
        M["VfiveNumCertImages"] = f"{ni:,}".replace(",", "{,}")
        M["VfiveZeroFailInj"] = f"{100 * zero_fail_bound(nc):.2f}\\%"
        M["VfiveZeroFailImg"] = f"{100 * zero_fail_bound(ni):.2f}\\%"
        say(f"   certified (P3) by the certificate: {nc} decisions on {ni} images; 95% bound "
            f"{M['VfiveZeroFailInj']} per decision, {M['VfiveZeroFailImg']} per image; "
            f"plus {nstruct} faults harmless by structure (non-class rows after the last attention)")
        # recompute share on harmless faults, P0
        det = [r for r in R if r["alarm_P0"] == "True"]
        if det:
            M["VfiveHarmlessDetectedPct"] = fmt(100 * np.mean([r["changed"] != "True" for r in det]))
        # Clopper-Pearson intervals for the main rates
        for p in cnt:
            k = cnt[p].get("recompute", 0)
            lo, hi = clopper_pearson(k, n)
            M[f"Vfive{p}RecPct"] = fmt(100 * k / n)
            M[f"Vfive{p}RecCI"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        # relaxed empirical thresholds (ApproxABFT/AdaptABFT principle)
        sites = sorted({int(r["site"]) for r in R})
        for bud, tag in ((0.0, "Zero"), (0.01, "One")):
            out, p0, kap, test = empirical_baseline(R, sites, bud)
            M[f"VfiveEmp{tag}Recompute"] = str(out["recompute"])
            M[f"VfiveEmp{tag}SilentWrong"] = str(out["silent_wrong"])
            M[f"VfiveEmp{tag}N"] = str(out["n"])
            M[f"VfiveEmp{tag}PzeroRecompute"] = str(p0["recompute"])
            M[f"VfiveEmp{tag}PzeroSilentWrong"] = str(p0["silent_wrong"])
            p3 = dict(recompute=sum(r["out_P3"] == "recompute" for r in test),
                      silent_wrong=sum(r["out_P3"] == "silent_wrong" for r in test),
                      wrong=sum(r["out_P3"] == "CERT_WRONG" for r in test))
            M[f"VfiveEmp{tag}PthreeRecompute"] = str(p3["recompute"])
            M[f"VfiveEmp{tag}PthreeSilentWrong"] = str(p3["silent_wrong"])
            say(f"   empirical thresholds (budget {bud:.0%}, tuned on half the images, {out['n']} test injections): "
                f"recompute {out['recompute']}, silent wrong {out['silent_wrong']}  |  P0g recompute "
                f"{p0['recompute']}, silent wrong {p0['silent_wrong']}  |  P3 recompute {p3['recompute']}, "
                f"silent wrong {p3['silent_wrong']}, certified wrong {p3['wrong']}")
        # figure: outcome shares per policy
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cats = ("recompute", "certified", "silent_ok", "silent_wrong")
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
            P = list(cnt)
            bottom = np.zeros(len(P))
            for cat in cats:
                v = np.array([cnt[p].get(cat, 0) for p in P]) / n * 100
                ax.bar(P, v, bottom=bottom, label=cat.replace("_", " "))
                bottom += v
            ax.set_ylabel("% of injections", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5.5, frameon=False, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(a.res, "fig_v5_policies.pdf"))
            plt.close(fig)
        except Exception as ex:
            say(f"(policy figure skipped: {ex})")

    # ---------------------------------------------------------------- E4
    p = os.path.join(a.res, "e4_faultmodels.csv")
    if os.path.exists(p):
        R = list(csv.DictReader(open(p)))
        say("E4 fault models:")
        rows = []
        for kind in ("acc", "oprow", "wcol", "shared", "double", "twosite"):
            K = [r for r in R if r["kind"] == kind]
            if not K:
                continue
            nk = len(K)
            pc = lambda f: 100.0 * sum(f(r) for r in K) / nk
            row = dict(kind=kind, status=K[0]["status"], n=nk,
                       masked=pc(lambda r: r["masked"] == "True"),
                       det_col=pc(lambda r: r["det_col"] == "True"),
                       det_two=pc(lambda r: r["det_two"] == "True"),
                       bod_col=sum(r["bod_col"] != "True" for r in K),
                       bod_two=sum(r["bod_two"] != "True" for r in K),
                       sw_col=sum(r["det_col"] != "True" and r["changed"] == "True" for r in K),
                       sw_two=sum(r["det_two"] != "True" and r["changed"] == "True" for r in K),
                       cert=sum(r["cert"] == "True" for r in K),
                       cert_wrong=sum(r["cert_wrong"] == "True" for r in K))
            rows.append(row)
            say(f"   {kind:8s} {row['status']:24s} n {nk:5d} masked {row['masked']:5.1f}% detected col "
                f"{row['det_col']:5.1f}% two-sided {row['det_two']:5.1f}%  BOD! col {row['bod_col']} "
                f"two-sided {row['bod_two']}  silent-wrong col {row['sw_col']} two-sided {row['sw_two']}  "
                f"certified {row['cert']} wrong {row['cert_wrong']}")
        with open(os.path.join(a.res, "table_e4.tex"), "w") as f:
            for r in rows:
                f.write(f"{r['kind']} & {r['status']} & {r['n']} & {r['masked']:.1f} & {r['det_col']:.1f} & "
                        f"{r['det_two']:.1f} & {r['bod_col']} & {r['bod_two']} & {r['sw_col']} & {r['sw_two']} \\\\\n")
        pa = os.path.join(a.res, "e4_audit.npz")
        if os.path.exists(pa):
            x = np.load(pa)["audit"]
            say(f"   E4 audit {int(x[0])}/{int(x[1])}")

    # ---------------------------------------------------------------- bench
    p = os.path.join(a.res, "bench_gpu.json")
    if os.path.exists(p):
        b = json.load(open(p))
        A = b.get("A", {})
        if A:
            say(f"GPU: {b.get('gpu')}")
            for k, v in A.items():
                if isinstance(v, dict) and "ms" in v:
                    bkey = k.split("/")[0] + "/emu+none"
                    t = v.get("graph_ms") or v["ms"]
                    base = A.get(bkey, {}).get("graph_ms") or A.get(bkey, {}).get("ms")
                    extra = f" (+{100 * (t / base - 1):.1f}% vs no checks)" if base and "emu+" in k and "none" not in k else ""
                    say(f"   {k:20s} graph {t:8.3f} ms  {1000.0 * int(k[1:k.index('/')]) / t:8.1f} img/s  "
                        f"work {v['mem_MB']:7.1f} MB{extra}")
            M["VfiveGPU"] = str(b.get("gpu", "")).replace("_", " ")
            for key, mac in (("B1/fp16", "LatFp"), ("B1/emu+none", "LatEmu"), ("B1/emu+col", "LatCol"),
                             ("B1/emu+grouped", "LatGrp"), ("B1/emu+twosided", "LatTwo")):
                if key in A:
                    M["Vfive" + mac] = fmt(A[key].get("graph_ms") or A[key]["ms"], 2)
            for bb in (64,):
                for key, mac in ((f"B{bb}/emu+none", "ThrEmu"), (f"B{bb}/emu+col", "ThrCol"),
                                 (f"B{bb}/emu+grouped", "ThrGrp"), (f"B{bb}/emu+twosided", "ThrTwo")):
                    if key in A:
                        M["Vfive" + mac] = fmt(A[key].get("graph_img_per_s") or A[key]["img_per_s"], 0)
        B = b.get("B", {})
        for bk in ("numpy", "cupy"):
            if f"total_ms_{bk}" in B:
                M[f"VfiveCert{bk.capitalize()}Ms"] = fmt(B[f"total_ms_{bk}"], 1)
                say(f"   certificate ({bk}): {B[f'total_ms_{bk}']:.1f} ms per inference for all enabled sites")
    with open(os.path.join(a.res, "macros_v5.tex"), "w") as f:
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(a.res, "summary_v5.txt"), "w") as f:
        f.write("\n".join(S) + "\n")
    print(f"wrote {a.res}/macros_v5.tex ({len(M)} macros) and summary_v5.txt")


if __name__ == "__main__":
    main()
