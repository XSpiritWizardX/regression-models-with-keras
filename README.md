# Regression Models with Keras (script version)

This repo runs the concrete strength regression example from the course without needing a notebook.

## Dataset
- Source: `concrete_data.csv` from the IBM CognitiveClass bucket.
- Predictors: Cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate.
- Target: Compressive strength.
- Predictors are normalized (zero mean, unit variance) before training.

## Setup
```bash
pipenv --python 3.11 install
pipenv shell  # or prefix commands with 'pipenv run'
```

## Usage
```bash
# Two hidden layers (default)
pipenv run python regression_models.py --model base --epochs 100 --val-split 0.3

# Five hidden layers
pipenv run python regression_models.py --model deep --epochs 150 --val-split 0.2

# Optional batch size
pipenv run python regression_models.py --batch-size 32
```

The script prints final loss/val_loss and the best validation loss epoch for quick iteration.

### Visualizations
Generate quick plots (saved under `figures/` by default):
```bash
pipenv run python visualize_data.py --output-dir figures
```
Creates histograms, a correlation heatmap, and predictor-vs-target scatter plots.

