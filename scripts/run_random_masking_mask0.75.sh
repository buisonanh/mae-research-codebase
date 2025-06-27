#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="random" MASK_RATIO=0.75 DEVICE="cuda:0" bash run_training.sh
