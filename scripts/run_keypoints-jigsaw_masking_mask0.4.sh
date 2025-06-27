#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="keypoints-jigsaw" MASK_RATIO=0.4 DEVICE="cuda:0" bash run_training.sh
