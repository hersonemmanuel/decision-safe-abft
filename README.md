# Decision-safe detection thresholds for ABFT

Code and results for the paper

> *Decision-Safe Detection Thresholds for Algorithm-Based Fault
> Tolerance in Low-Precision Transformer Inference.

The paper develops a theory of detection thresholds for algorithm-based fault
tolerance (ABFT): when an undetected fault is provably bounded, when a
threshold can be both free of false alarms and safe for a model's decision
(the certifiability window, which exists exactly when the decision budget is
at least twice the fault-free residual bound), and how to bound that
residual soundly for low-precision arithmetic. This repository contains
everything needed to reproduce the experiments.

## What is here

| file | purpose |
|---|---|
| `dcft/model.py` | DeiT-Tiny in NumPy: a float64 reference and an emulated accelerator with binary16 GEMMs, binary32 accumulation and column checksums at 49 sites |
| `dcft/certificate.py` | sound propagation rules and the per-site decision certificate |
| `dcft/arith.py` | rounding emulation, bit flips, path constants |
| `run_e1_bounds.py` | E1: soundness and tightness of the residual bound under four accumulator designs |
| `run_e2_calibrate.py` | E2: false alarms in situ and calibration of decision budgets |
| `run_e3_faults.py` | E3: single-bit-flip fault injection under three threshold policies |
| `make_results.py` | turns raw outputs into `results/macros.tex` and the figures used by the paper |
| `diagnose.py` | per-site breakdown of E3 |
| `verify_forward.py` | checks the NumPy model against timm in double precision |
| `self_test.sh` | one-minute check with random weights, no downloads |
| `results/` | the raw outputs reported in the paper |

The model is re-implemented in NumPy on purpose: every rounding step is
explicit, which the error bounds require. Everything runs on a laptop CPU.

## Reproducing the results

Python 3.9 or newer.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash self_test.sh            # must report 0 violations of every kind
```

Data: the ImageNetV2 matched-frequency test set (Recht et al., ICML 2019),
from https://huggingface.co/datasets/vaishaal/ImageNetV2 (file
`imagenetv2-matched-frequency.tar.gz`), extracted into `data/`. Please
follow the dataset's terms of use.

```bash
export IMAGES=data/imagenetv2-matched-frequency-format-val
python export_weights.py                         # DeiT-Tiny weights via timm
python verify_forward.py --images "$IMAGES" --n 5  # must end with "-> OK"
bash run_all.sh
```

On one CPU core: E1 about 10 minutes, E2 about 30 minutes, E3 about 40
minutes. Image order is fixed by a seeded shuffle, and all random choices use
fixed seeds, so the runs are repeatable on the same platform. Results can
differ in the last digits across BLAS libraries, because the summation order
of the platform GEMM is not specified; the bounds use the worst-case order
and hold for any of them.

## Built-in checks

Each experiment checks the theory on the fly and prints a summary:

* E1: observed residuals above the proved bound (`violations`, expected 0);
* E2: false alarms of the sound thresholds on fault-free runs (expected 0);
* E3: undetected faults larger than the proved bound
  (`bounded-or-detected violations`), certified decisions that differ from
  the fault-free run (`certified-wrong`), and a stage-by-stage audit of the
  certificate against the fault-free run (`audit violations`), all expected 0.

## Licence and citation

MIT licence (see `LICENSE`). If you use this code, please cite the paper
(see `CITATION.cff`). DeiT weights and ImageNetV2 are third-party resources
under their own terms and are not redistributed here.
