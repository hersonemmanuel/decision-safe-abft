#!/usr/bin/env bash
# Full pipeline. Set IMAGES to the extracted ImageNetV2 folder, e.g.
#   IMAGES=data/imagenetv2-matched-frequency-format-val bash run_all.sh
set -e
IMAGES=${IMAGES:-data/imagenetv2-matched-frequency-format-val}
W=weights/deit_tiny.npz
python export_weights.py
python verify_forward.py --images "$IMAGES" --n 5
python run_e1_bounds.py    --weights $W --images "$IMAGES" --n-images 50 --tiles-per-image 8
python run_e2_calibrate.py --weights $W --images "$IMAGES" --n-cal 200 --n-test 300
python run_e3_faults.py    --weights $W --images "$IMAGES" --n-images 300 --inj-per-image 30
python make_results.py
python diagnose.py
