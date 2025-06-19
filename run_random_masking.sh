#!/bin/bash
source .venv/bin/activate
MASKING_STRATEGY="random" DEVICE="cuda:0" bash run_training.sh
