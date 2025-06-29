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
    # Use tee to show output in terminal while capturing it
    PRETRAIN_OUTPUT=$(python3 -m src.pretrain 2>&1 | tee /dev/tty)
    PRETRAIN_STATUS=${PIPESTATUS[0]}

    if [ $PRETRAIN_STATUS -ne 0 ]; then
        echo "Error: Pretraining failed!"
        echo "Output from pretraining script:"
        echo "$PRETRAIN_OUTPUT"
        exit 1
    fi

    # Extract the checkpoint path from the output
    CHECKPOINT_FROM_PRETRAIN=$(echo "$PRETRAIN_OUTPUT" | grep "Best model saved at:" | awk '{print $5}')
    echo "Pretraining completed successfully!"
fi

# Extract SAVE_PATH from config.py using Python
SAVE_PATH=$(python -c "from src.config import SAVE_PATH; print(SAVE_PATH)")
CHECKPOINT_PATH="$SAVE_PATH/pretrain_checkpoints/mae_checkpoints/best_model.pth"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$CHECKPOINT_PATH
fi

# Run classification if not skipped
if $CLASSIFY; then
    echo "Starting classification"

    # Prioritize user-provided checkpoint
    if [ -n "$CHECKPOINT" ]; then
        echo "Using user-provided checkpoint: $CHECKPOINT"
        python3 -m src.classify --checkpoint "$CHECKPOINT"
    # Otherwise, use the checkpoint from the pretraining step if it exists
    elif [ -n "$CHECKPOINT_FROM_PRETRAIN" ]; then
        echo "Using checkpoint from pretraining: $CHECKPOINT_FROM_PRETRAIN"
        python3 -m src.classify --checkpoint "$CHECKPOINT_FROM_PRETRAIN"
    # Otherwise, run without a specific checkpoint
    else
        echo "No checkpoint specified. Running classification without a pretrained model."
        python3 -m src.classify
    fi

    if [ $? -ne 0 ]; then
        echo "Error: Classification failed!"
        exit 1
    fi
    echo "Classification completed successfully!"
fi