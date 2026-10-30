"""Soundness test of the relational certificate: random faults, audit of
every stage against the golden run. Expected: 0 audit violations."""
import argparse, time
import numpy as np
from dcft.model import Weights
from dcft.faultsim import hw_forward_g, hw_forward_faulty_g, gemm_of, theta_rows
from dcft.cert_rel import RelCert
from common import image_source

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/synthetic.npz")
ap.add_argument("--images", default=None)
ap.add_argument("--synthetic", type=int, default=1)
ap.add_argument("--n-images", type=int, default=2)
ap.add_argument("--inj", type=int, default=20)
ap.add_argument("--blocks", default="10,11")
ap.add_argument("--gsize", type=int, default=-1)
ap.add_argument("--kind", default="acc")
ap.add_argument("--small", type=int, default=1, help="use small bit positions so faults are small")
a = ap.parse_args()
gs = None if a.gsize < 0 else a.gsize
w = Weights(a.weights)
blocks = [int(x) for x in a.blocks.split(",")]
sites = [i for i, (b, s) in enumerate(w.sites) if b in blocks] + [len(w.sites) - 1]
rng = np.random.default_rng(3)
tot_bad = tot_chk = 0; npass = ntry = 0; times = []
for k, (img, _) in enumerate(image_source(a.images, a.synthetic, 0, a.n_images)):
    clean = hw_forward_g(w, img, gs)
    for _ in range(a.inj):
        si = int(rng.choice(sites)); b, s = w.sites[si]
        g0 = gemm_of(clean, (b, s), w.depth)
        m, N = g0["acc"].shape
        t = 0 if rng.random() < 0.3 else int(rng.integers(m))
        if a.kind == "acc":
            j = int(rng.integers(N))
            bit = int(rng.integers(0, 23 if a.small else 32))
            f = dict(kind="acc", block=b, stage=s, row=t, col=j, bit=bit)
        else:   # operand row fault
            j = -1
            kk = int(rng.integers(w.lin((b, s)).K))
            f = dict(kind="oprow", block=b, stage=s, row=t, k=kk, bit=int(rng.integers(0, 10)))
        fr = hw_forward_faulty_g(w, clean, [f], gs)
        g = gemm_of(fr, (b, s), w.depth)
        th = theta_rows(g)
        if not np.all(np.isfinite(th)):
            continue
        if j >= 0 and abs(float(g["acc"][t, j]) - float(g0["acc"][t, j])) > th[t, j]:
            print("BOD violated?!"); continue
        log = []
        aud = dict(run=fr, golden=clean, t=t, j=j, log=log)
        t0 = time.perf_counter()
        rc = RelCert(w, fr, first_block=min(blocks), audit=aud)
        ok = rc.certify(si, th, "entry" if a.kind == "acc" else "row")
        times.append(time.perf_counter() - t0)
        bad = sum(v for _, v, _ in log); chk = sum(n for _, _, n in log)
        tot_bad += bad; tot_chk += chk; ntry += 1; npass += ok
        if bad:
            print("VIOLATION", w.sites[si], t, j, [(n, v) for n, v, _ in log if v][:5])
        dec_ok = int(np.argmax(fr["head"]["z"])) == int(np.argmax(clean["head"]["z"]))
        if ok and not dec_ok:
            print("CERTIFIED WRONG", w.sites[si])
print(f"tries {ntry}, certified {npass}, audit violations {tot_bad}/{tot_chk}, "
      f"median time {np.median(times):.3f}s, max {np.max(times):.3f}s")
