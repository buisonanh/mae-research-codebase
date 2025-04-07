# MAE Research Codebase (CNN)

This repository contains the implementation for MAE research using CNN models.

## Setup and Usage Instructions

1. **Install Dependencies**
   ```bash
   uv sync
   ```

2. **Download Datasets**
   Run the dataset download script:
   ```bash
   ./download_dataset.sh --name <dataset_name>
   ```
   Replace `<dataset_name>` with the desired dataset name.

3. **Configure Settings**
   Adjust the configuration settings in `config.py`:
   ```bash
   nano src/config.py
   ```
   Modify the parameters according to your needs.

4. **Start Training**
   Begin the training process by running:
   ```bash
   ./run_training.sh
   ```

## Project Structure
```
mae-research-codebase-cnn/
├── src/
│   ├── config.py          # Configuration settings
│   ├── pretrain.py        # Pretraining module
│   ├── classify.py        # Classification module
│   └── models/
│   │   ├── autoencoder.py  # Autoencoder model
│   │   ├── classifier.py   # Classifier model
│   │   └── keypoint_model.py  # Keypoint detection model
│   └── utils/
│   │   ├── train_keypoints.py  # Keypoint training utilities
│   │   ├── visualization.py    # Visualization utilities
│   │   └── masking.py         # Masking utilities
│   └── data/
│       └── dataset.py         # Dataset utilities
├── download_dataset.sh    # Dataset download script
└── run_training.sh       # Training script
```

## Requirements
- Python 3.x
- Dependencies managed via uv (uv sync)