#!/bin/bash

# Activate conda environment (uncomment and modify if needed)
# conda activate mae-research

# Set CUDA device (modify as needed)
# export CUDA_VISIBLE_DEVICES=2

# Parse command line arguments
PRETRAIN=true
CLASSIFY=true
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-pretrain)
      PRETRAIN=false
      shift
      ;;
    --skip-classify)
      CLASSIFY=false
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--skip-pretrain] [--skip-classify] [--checkpoint PATH]"
      exit 1
      ;;
  esac
done

# Run pretraining if not skipped
if $PRETRAIN; then
    echo "Starting MAE pretraining..."
    python -m src.train
    
    # Check if pretraining was successful
    if [ $? -ne 0 ]; then
        echo "Error: Pretraining failed!"
        exit 1
    fi
    echo "Pretraining completed successfully!"
else
    echo "Skipping pretraining phase..."
fi

# Run classification if not skipped
if $CLASSIFY; then
    echo "Starting classification training..."
    
    # Run classification training with or without checkpoint
    if [ -n "$CHECKPOINT" ]; then
        echo "Using pretrained checkpoint: $CHECKPOINT"
        python -m src.classify --checkpoint "$CHECKPOINT"
    else
        echo "Using default pretrained weights"
        python -m src.classify
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Classification training failed!"
        exit 1
    fi
    echo "Classification training completed successfully!"
else
    echo "Skipping classification phase..."
fi

echo "All requested training steps completed successfully!"