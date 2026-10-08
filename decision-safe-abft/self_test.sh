#!/usr/bin/env bash
# Five-minute check with random weights and random images (no downloads).
set -e
python make_synthetic_weights.py weights/synthetic.npz
python run_e1_bounds.py   --weights weights/synthetic.npz --synthetic 1 --n-images 2 --tiles-per-image 2 --out results_selftest
python run_e2_calibrate.py --weights weights/synthetic.npz --synthetic 1 --n-cal 3 --n-test 2 --out results_selftest
python run_e3_faults.py   --weights weights/synthetic.npz --synthetic 1 --calib results_selftest/calib.json --n-images 2 --inj-per-image 20 --audit-every 1 --out results_selftest
python make_results.py --res results_selftest
echo "Self-test finished: check that 'certified-wrong 0', 'bounded-or-detected violations 0' and 'audit violations 0' were printed above."
