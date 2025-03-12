#!/bin/bash

# Activate conda environment (uncomment and modify if needed)
# conda activate mae-research

# Set CUDA device (modify as needed)
# export CUDA_VISIBLE_DEVICES=2

echo "Starting MAE pretraining..."

# Run pretraining
python src/train.py

# Check if pretraining was successful
if [ $? -eq 0 ]; then
    echo "Pretraining completed successfully!"
    echo "Starting classification training..."
    
    # Run classification training
    python src/classify.py
    
    if [ $? -eq 0 ]; then
        echo "Classification training completed successfully!"
    else
        echo "Error: Classification training failed!"
        exit 1
    fi
else
    echo "Error: Pretraining failed!"
    exit 1
fi

echo "All training completed successfully!" 