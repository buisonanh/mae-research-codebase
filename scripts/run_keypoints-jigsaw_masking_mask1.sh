#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="keypoints-jigsaw" MASK_RATIO=1 DEVICE="cuda:0" bash run_training.sh --
