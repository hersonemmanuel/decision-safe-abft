"""Per-site breakdown of E3, to see where the certificate passes and fails."""
import csv, collections, json, sys
import numpy as np

res = sys.argv[1] if len(sys.argv) > 1 else "results"
R = list(csv.DictReader(open(f"{res}/e3_injections.csv")))
calib = json.load(open(f"{res}/calib.json"))
ts = calib["theta_sites"]
print("sites with a calibrated budget:")
for i, t in enumerate(ts):
    if t > 0:
        rows = [r for r in R if int(r["site"]) == i]
        und = [r for r in rows if r["alarm_P1"] == "False"]
        cp = [r for r in und if r["cert_P1"] == "True"]
        ob = [float(r["theta_obs"]) for r in und if r["theta_obs"] not in ("", None)]
        print(f"  site {i} ({calib['sites'][i]}): theta_s={t:.3g}  injections={len(rows)}  "
              f"undetected={len(und)}  certified={len(cp)}  fallback={len(und) - len(cp)}  "
              f"det_P0={sum(r['alarm_P0'] == 'True' for r in rows)}  det_P1={sum(r['alarm_P1'] == 'True' for r in rows)}")
        if ob:
            q = np.percentile(ob, [0, 50, 90, 100])
            print(f"      theta_obs  min/median/90%/max = {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g} / {q[3]:.3g}"
                  f"   fraction above theta_s: {np.mean(np.array(ob) > t):.2f}")
        fb = [r for r in und if r["cert_P1"] != "True"]
        if fb:
            m = [float(r["margin_gold"]) for r in fb]
            print(f"      golden margin of fallback cases: median {np.median(m):.3g}, min {min(m):.3g}")
print("\nsilent decision changes under P0:")
for r in R:
    if r["out_P0"] == "silent_wrong":
        print(f"  image {r['image']} site {r['site']} ({r['block']}.{r['stage']}) bit {r['bit']} "
              f"|delta|/eta={float(r['delta']) / float(r['eta_col']):.3g} margin={float(r['margin_gold']):.3g}")
