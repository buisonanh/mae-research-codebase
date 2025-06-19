#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="keypoints-jigsaw" DEVICE="cuda:2" bash run_training.sh
