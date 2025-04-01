# **MAE Research Codebase**

## **Installation**

1. Clone the repository:
```bash
git clone https://github.com/sonanhbui/mae-research-codebase.git
```

2. Install dependencies (using uv):
If you don't have uv installed, you can install it using pip:
```bash
pip install uv
```

Then, install dependencies:
```bash
uv sync
```

3. Download datasets:
```bash
./download_dataset.sh --name rafdb
./download_dataset.sh --name affectnet
```

## **Training**

1. Set config parameters in `src/config.py`:

2. Run training both pretraining and classification:
```bash
./run_training.sh
```



