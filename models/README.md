# Models Directory

This directory contains trained YOLO models and champion model artifacts.

## Structure

```
models/
├── champion/                    # Current best performing model
│   ├── model/                  # Model artifacts from MLflow
│   │   ├── best.pt            # Best model weights
│   │   └── last.pt            # Last epoch weights
│   └── champion_info.json     # Champion model metadata
├── train_20241023_140532/      # Individual training run outputs
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png            # Training curves
│   └── confusion_matrix.png   # Validation metrics
└── train_20241023_151045/      # Another training run
    └── ...
```

## Champion Model

The `champion/` directory contains:

- **Model artifacts**: Best performing model weights
- **Metadata**: Training information, metrics, and timestamps
- **Configuration**: Dataset and training parameters used

## Training Run Outputs

Each training run creates a timestamped directory with:

- Model weights (best.pt, last.pt)
- Training curves and visualizations
- Validation results
- Configuration files

## Cleanup Policy

- Keeps the last 5 training runs by default
- Older runs are automatically cleaned up
- Champion model is never deleted
- Manual cleanup available via CLI

## Usage

```bash
# List all models
ls -la models/

# Check champion model info
cat models/champion/champion_info.json

# Load champion model in Python
from ultralytics import YOLO
model = YOLO("models/champion/model/best.pt")
```