#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="combined-keypoints-jigsaw-random-mask" DEVICE="cuda:2" bash run_training.sh
