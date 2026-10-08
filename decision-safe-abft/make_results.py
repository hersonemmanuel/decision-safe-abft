"""Turn the raw outputs of E1-E3 into results/macros.tex and figures.

The paper reads every number from results/macros.tex, so nothing in the
text is typed by hand. Missing experiments leave their macros undefined and
the paper prints a visible placeholder instead.
"""
import argparse
import csv
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 150})


def fmt(x, digits=3):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", "{,}")
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        m, e = f"{x:.2e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{x:.{digits}g}"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    R = a.res
    M = {}

    # ---------------- E1 ----------------
    p = os.path.join(R, "e1_tiles.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        rat = np.load(os.path.join(R, "e1_ratios.npz"))
        names = {("fp32", "seq"): "FpSeq", ("fp32", "tree"): "FpTree",
                 ("bf16", "seq"): "BfSeq", ("bf16", "tree"): "BfTree"}
        tiles = len(rows) // 4
        M["EoneTiles"] = fmt(tiles)
        M["EoneViolations"] = fmt(sum(int(r["violations"]) for r in rows))
        for (f, o), nm in names.items():
            sub = [r for r in rows if r["fmt"] == f and r["order"] == o]
            v = rat[f"{f}_{o}"]
            cols = sum(int(r["cols"]) for r in sub)
            M["EoneCols"] = fmt(cols)
            M["EoneMax" + nm] = fmt(float(v.max()))
            M["EoneMed" + nm] = fmt(float(np.median(v)))
            M["EoneRel" + nm] = fmt(pct(sum(int(r["fa_rel"]) for r in sub), cols), 2)
            M["EoneLin" + nm] = fmt(sum(int(r["fa_lin"]) for r in sub))
        i8 = rat["int8"]
        M["EoneIntZero"], M["EoneIntTotal"] = fmt(int(i8[0])), fmt(int(i8[1]))
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        for (f, o), nm in names.items():
            v = rat[f"{f}_{o}"]
            v = v[v > 0]
            ax.hist(np.log10(v), bins=60, histtype="step", label=f"{f.upper()} {o}")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(r"$\log_{10}(|r_j|/\eta_j)$")
        ax.set_ylabel("columns")
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_e1.pdf"))

    # ---------------- E2 ----------------
    enabled = None
    p = os.path.join(R, "e2.npz")
    if os.path.exists(p):
        d = np.load(p)
        calib = json.load(open(os.path.join(R, "calib.json")))
        depth = int(d["depth"])
        TH, WR = d["theta"], d["window_ratio"]
        M["NumCal"], M["NumTest"] = fmt(TH.shape[0]), fmt(len(d["agree"]))
        M["QuantileLevel"] = fmt(float(calib["q"]))
        M["FalseAlarms"], M["NumChecks"] = fmt(int(d["false_alarms"])), fmt(int(d["n_checks"]))
        M["InSituMaxRatio"] = fmt(float(d["ratio_max"].max()))
        M["HwRefAgree"] = fmt(pct(int(d["agree"].sum()), len(d["agree"])), 4)
        if d["top1"].shape[0]:
            M["TopOneHw"] = fmt(100 * d["top1"][:, 0].mean(), 3)
            M["TopOneRef"] = fmt(100 * d["top1"][:, 1].mean(), 3)
        ts = d["theta_sites"]
        M["SitesCertified"] = fmt(int(np.sum(ts > 0)))
        M["SitesTotal"] = fmt(len(ts))
        M["TimeFwd"] = fmt(1000 * float(np.median(d["t_fwd"])), 3)
        M["TimeCal"] = fmt(float(np.median(d["t_cal"])), 3)
        nonempty = WR >= 1
        M["WindowNonemptyAll"] = fmt(100 * nonempty.mean(), 3)
        last = list(range(4 * (depth - 1), 4 * depth + 1))
        M["WindowNonemptyLast"] = fmt(100 * nonempty[:, last].mean(), 3)
        # deepest-first block from which windows open for half the inputs
        frac = nonempty.mean(axis=0)
        M["WindowHalfSites"] = fmt(int(np.sum(frac >= 0.5)))
        # a site is enabled for relaxed thresholds only if its window is open
        # for at least a (1-q) share of the calibration inputs
        q = float(calib["q"])
        enabled = (np.quantile(WR, q, axis=0) >= 1.0) & (ts > 0)
        M["SitesEnabled"] = fmt(int(enabled.sum()))
        names = [calib["sites"][i] for i in np.where(enabled)[0]]
        M["EnabledSiteNames"] = r"\text{" + ", ".join(n.replace("12.head", "head") for n in names) + "}" if names else r"\text{none}"
        # distance to the window: log10 of theta*/(2 eta) style ratio, per site
        floor = -12.0
        L = np.log10(np.maximum(WR, 10.0 ** floor))
        med = np.median(L, axis=0)
        lo, hi = np.percentile(L, 5, axis=0), np.percentile(L, 95, axis=0)
        M["MedianGapEarly"] = fmt(float(np.median(med[:4 * (depth - 1)])))
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        x = np.arange(len(med))
        ax.fill_between(x, lo, hi, color="0.85", lw=0)
        ax.plot(x, med, "o-", ms=2, color="0.15", lw=0.8)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("checked GEMM (block order; last point = head)")
        ax.set_ylabel(r"$\log_{10}$ window ratio")
        ax.set_ylim(floor - 0.5, max(1.0, float(hi.max()) + 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_e2.pdf"))

    # ---------------- E3 ----------------
    p = os.path.join(R, "e3_injections.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        n = len(rows)
        tb = lambda v: v == "True"
        if enabled is not None:
            # sites with a positive budget whose window stays closed
            closed = [i for i in range(len(enabled)) if (not enabled[i]) and
                      any(int(r["site"]) == i and float(r["theta_site"]) > 0 for r in rows)]
            if closed:
                i0 = closed[0]
                ob = [float(r["theta_obs"]) for r in rows
                      if int(r["site"]) == i0 and r.get("theta_obs") not in ("", None)]
                M["ClosedSiteName"] = r"\text{" + json.load(open(os.path.join(R, "calib.json")))["sites"][i0] + "}"
                M["ClosedSiteTheta"] = fmt(float(next(r["theta_site"] for r in rows if int(r["site"]) == i0)))
                if ob:
                    M["ClosedSiteObs"] = fmt(float(np.median(ob)))
            # sites that do not meet the window rule keep the P0 thresholds
            for r in rows:
                if not enabled[int(r["site"])]:
                    r["out_P1"] = r["out_P0"]
                    r["out_P2"] = r["out_P0"]
                    r["alarm_P1"] = r["alarm_P0"]
                    r["cert_P1"] = ""
                    r["theta_site"] = "0"
        M["NumInj"] = fmt(n)
        M["MaskedPct"] = fmt(pct(sum(tb(r["masked"]) for r in rows), n), 3)
        for pol, nm in (("P0", "Pzero"), ("P1", "Pone"), ("P2", "Ptwo")):
            c = collections.Counter(r[f"out_{pol}"] for r in rows)
            for key, kn in (("detected", "Det"), ("silent_ok", "SilentOk"), ("silent_wrong", "SilentWrong"),
                            ("certified", "Cert"), ("fallback", "Fallback"), ("CERT_WRONG", "CertWrong")):
                M[nm + kn] = fmt(int(c.get(key, 0)))
        M["BodViolations"] = fmt(sum(not (tb(r["bod_P0"]) and tb(r["bod_P1"])) for r in rows))
        au = np.load(os.path.join(R, "e3_audit.npz"))["audit"]
        M["AuditViol"], M["AuditChecks"] = fmt(int(au[0])), fmt(int(au[1]))
        cs = [r for r in rows if float(r["theta_site"]) > 0]
        und = [r for r in cs if not tb(r["alarm_P1"])]
        certified = [r for r in und if r["cert_P1"] == "True"]
        M["NumCertSiteInj"], M["NumCertSiteUndet"] = fmt(len(cs)), fmt(len(und))
        M["NumCertifiedCases"] = fmt(len(certified))
        ncert = len(certified)
        M["ZeroFailBound"] = fmt(1 - 0.05 ** (1 / ncert)) if ncert else "--"
        rec0 = sum(tb(r["alarm_P0"]) for r in cs)
        rec1 = sum(tb(r["alarm_P1"]) or r["out_P1"] == "fallback" for r in cs)
        M["RecomputePzeroCS"], M["RecomputePoneCS"] = fmt(rec0), fmt(rec1)
        M["UndetPzeroCS"] = fmt(len(cs) - rec0)
        M["RecomputeReduction"] = fmt(pct(rec0 - rec1, rec0), 3) if rec0 else "0"
        M["RecomputeChange"] = fmt(pct(rec1 - rec0, rec0), 3) if rec0 else "0"
        det0 = [r for r in rows if tb(r["alarm_P0"]) and np.isfinite(float(r["delta"]))]
        M["HarmlessDetectedPct"] = fmt(pct(sum(not tb(r["changed"]) for r in det0), len(det0)), 3)
        ch = [float(r["delta"]) / float(r["eta_col"]) for r in rows
              if tb(r["changed"]) and np.isfinite(float(r["delta"])) and float(r["eta_col"]) > 0]
        M["MinChangeRatio"] = fmt(min(ch)) if ch else "--"
        sw0 = [r for r in rows if r["out_P0"] == "silent_wrong"]
        if sw0:
            tie = max(float(r["margin_gold"]) for r in sw0)
            chr_ = [float(r["delta"]) / float(r["eta_col"]) for r in rows
                    if tb(r["changed"]) and np.isfinite(float(r["delta"])) and float(r["eta_col"]) > 0
                    and float(r["margin_gold"]) > tie]
            M["MinChangeRatioRobust"] = fmt(min(chr_)) if chr_ else "--"
            M["NumTieImages"] = fmt(len({r["image"] for r in sw0}))
        sw = [r for r in rows if r["out_P0"] == "silent_wrong"]
        M["SilentWrongBelowTwoEta"] = fmt(sum(float(r["delta"]) <= 2 * float(r["eta_col"]) for r in sw))
        if sw and "margin_gold" in rows[0]:
            mm = max(float(r["margin_gold"]) for r in sw)
            M["SilentWrongMaxMargin"] = fmt(mm)
            imgs = {}
            for r in rows:
                imgs[r["image"]] = float(r["margin_gold"])
            M["SmallMarginPct"] = fmt(pct(sum(v <= mm for v in imgs.values()), len(imgs)), 3)
            M["NumImagesEthree"] = fmt(len(imgs))
        M["CertSiteSilentWrongPzero"] = fmt(sum(r["out_P0"] == "silent_wrong" for r in cs))
        M["CertSiteSilentWrongPone"] = fmt(sum(r["out_P1"] == "silent_wrong" for r in cs))
        M["CertifiedShare"] = fmt(pct(len(certified), len(und)), 3) if und else "0"
        M["NumChanged"] = fmt(len(ch))
        # figure: probability of a decision change versus fault size relative to eta
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
        xs = np.array([float(r["delta"]) / float(r["eta_col"]) for r in rows
                       if np.isfinite(float(r["delta"])) and float(r["delta"]) > 0 and float(r["eta_col"]) > 0])
        ys = np.array([tb(r["changed"]) for r in rows
                       if np.isfinite(float(r["delta"])) and float(r["delta"]) > 0 and float(r["eta_col"]) > 0])
        if xs.size:
            bins = np.arange(np.floor(np.log10(xs.min())), np.ceil(np.log10(xs.max())) + 1)
            idx = np.digitize(np.log10(xs), bins)
            px, py = [], []
            for k in range(1, len(bins)):
                sel = idx == k
                if sel.sum() >= 5:
                    px.append(bins[k - 1] + 0.5)
                    py.append(100 * ys[sel].mean())
            axs[0].plot(px, py, "o-", ms=2.5, color="0.2")
            axs[0].axvline(np.log10(2.0), color="k", ls="--", lw=0.8)
        axs[0].set_xlabel(r"$\log_{10}(|\delta|/\eta_j)$")
        axs[0].set_ylabel("decision changed (%)")
        labels = ["P0", "P1", "P2"]
        cats = ["detected", "certified", "fallback", "silent_ok", "silent_wrong"]
        colors = ["0.25", "0.55", "0.75", "0.9", "black"]
        bottom = np.zeros(3)
        for cat, colr in zip(cats, colors):
            vals = np.array([pct(sum(r[f"out_{p}"] == cat for r in cs), len(cs)) for p in labels]) if cs else np.zeros(3)
            axs[1].bar(labels, vals, bottom=bottom, color=colr, edgecolor="k", lw=0.3, label=cat.replace("_", " "))
            bottom += vals
        axs[1].set_ylabel("injections at enabled sites (%)")
        axs[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(R, "fig_e3.pdf"))

    with open(os.path.join(R, "macros.tex"), "w") as f:
        f.write("% generated by make_results.py -- do not edit by hand\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{\\ensuremath{{{v}}}}}\n")
    print(f"wrote {len(M)} macros to {R}/macros.tex")


if __name__ == "__main__":
    main()
