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
    # Capture the output of the pretraining script to get the checkpoint path
    PRETRAIN_OUTPUT=$(python -m src.pretrain)
    
    if [ $? -ne 0 ]; then
        echo "Error: Pretraining failed!"
        echo "Output from pretraining script:"
        echo "$PRETRAIN_OUTPUT"
        exit 1
    fi

    # Extract the checkpoint path from the output
    CHECKPOINT_FROM_PRETRAIN=$(echo "$PRETRAIN_OUTPUT" | grep "Best model saved at:" | awk '{print $5}')
    echo "Pretraining completed successfully!"
fi

# Run classification if not skipped
if $CLASSIFY; then
    echo "Starting classification"
    
    # Prioritize user-provided checkpoint
    if [ -n "$CHECKPOINT" ]; then
        echo "Using user-provided checkpoint: $CHECKPOINT"
        python -m src.classify --checkpoint "$CHECKPOINT"
    # Otherwise, use the checkpoint from the pretraining step if it exists
    elif [ -n "$CHECKPOINT_FROM_PRETRAIN" ]; then
        echo "Using checkpoint from pretraining: $CHECKPOINT_FROM_PRETRAIN"
        python -m src.classify --checkpoint "$CHECKPOINT_FROM_PRETRAIN"
    # Otherwise, run without a specific checkpoint
    else
        echo "No checkpoint specified. Running classification without a pretrained model."
        python -m src.classify
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Classification failed!"
        exit 1
    fi
    echo "Classification completed successfully!"
fi