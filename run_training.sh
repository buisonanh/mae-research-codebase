#!/bin/bash

# Default values
PRETRAIN=true
CLASSIFY=true
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-pretrain)
      PRETRAIN=false
      shift
      ;;
    --no-classify)
      CLASSIFY=false
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-pretrain] [--no-classify] [--checkpoint PATH]"
      exit 1
      ;;
  esac
done

# Run pretraining if not skipped
if $PRETRAIN; then
    echo "Starting MAE pretraining"
    python -m src.pretrain
    
    if [ $? -ne 0 ]; then
        echo "Error: Pretraining failed!"
        exit 1
    fi
    echo "Pretraining completed successfully!"
fi

# Run classification if not skipped
if $CLASSIFY; then
    echo "Starting classification"
    if [ -n "$CHECKPOINT" ]; then
        echo "Using checkpoint: $CHECKPOINT"
        python -m src.classify --checkpoint "$CHECKPOINT"
    else
        echo "No pretrain checkpoint provided. Using non-pretrained model."
        python -m src.classify
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Classification failed!"
        exit 1
    fi
    echo "Classification completed successfully!"
fi