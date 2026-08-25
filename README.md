# Regression Models with Keras

A TensorFlow/Keras regression project for predicting **concrete compressive strength** from mixture composition and age. The project started from the IBM/CognitiveClass deep-learning exercise and was converted from a notebook-style walkthrough into a reusable Python/CLI project for model experimentation, visualization, and agent-assisted engineering.

## Why this project exists

This repository has two goals:

1. Demonstrate practical AI/ML engineering with TensorFlow and Keras.
2. Demonstrate an **agentic coding workflow** around the model: coding agents inspect the pipeline, identify risks, propose changes and experiments, help implement improvements, and surface evidence that is then reviewed against explicit quality gates.

The neural network itself is **not** an autonomous agent. The agentic component is the software-development and experimentation workflow around the model.

## Dataset

- Source: `concrete_data.csv` from the IBM CognitiveClass bucket.
- Predictors: Cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and **Age**.
- Target: Concrete compressive strength.
- Current implementation standardizes predictors to zero mean and unit variance before training.

## Model architectures

The project exposes two feed-forward regression architectures:

### Base model

```text
8 inputs -> Dense(50, ReLU) -> Dense(50, ReLU) -> Dense(1)
```

### Deep model

```text
8 inputs -> Dense(50, ReLU) -> Dense(50, ReLU) -> Dense(50, ReLU)
         -> Dense(50, ReLU) -> Dense(50, ReLU) -> Dense(1)
```

Both models use:

- TensorFlow/Keras `Sequential`
- ReLU hidden activations
- Linear scalar output for continuous regression
- Adam optimizer
- Mean squared error (MSE) loss

The architecture is parameterized through `build_model()`, so the experiment pipeline does not have to be rewritten when changing depth or hidden-unit configuration.

## Project structure

```text
regression_models.py   # data loading, normalization, model construction, training, CLI
visualize_data.py      # histograms, correlation heatmap, predictor/target scatter plots
requirements.txt       # runtime dependencies
Pipfile                # Python environment configuration
README.md              # project, ML methodology, and agentic workflow documentation
```

## Agentic AI development workflow

This repository is maintained with an **AI coding agent as an engineering collaborator**, not as an unquestioned code generator.

A typical iteration follows this loop:

```text
1. OBSERVE
   Inspect the current data pipeline, model architecture, metrics, code, and experiment output.

2. CRITIQUE
   Ask the coding agent to identify ML methodology risks, implementation defects,
   reproducibility problems, and opportunities for stronger evaluation.

3. PROPOSE
   Produce a bounded change or experiment with an explicit hypothesis and expected evidence.

4. IMPLEMENT
   Make the code/documentation change on a dedicated Git branch tied to a GitHub issue.

5. RUN / VERIFY
   Execute the model or relevant checks and collect objective evidence such as loss,
   validation loss, test metrics, plots, diffs, or test results.

6. ASSESS
   Compare the evidence with the hypothesis. A suggestion is not accepted simply because
   the agent generated working code.

7. RETAIN OR REJECT
   Keep the change only when the evidence and code review support it; otherwise revise or revert.

8. RECORD
   Preserve the reasoning and result in GitHub issues, commits, pull requests, and README notes.
```

This README expansion is itself an example of that workflow: **GitHub issue #1** was created first, the repository and current ML implementation were inspected by an AI coding agent, the documentation change was made on a dedicated branch, and the work was delivered through a pull request before merge.

## How agent output is evaluated

Agent-generated suggestions are assessed using software-engineering and ML-specific evidence.

### Code quality gates

- Does the change preserve a clear separation between data loading, model construction, training, and visualization?
- Is the behavior configurable rather than unnecessarily hard-coded?
- Does the change avoid silently altering the meaning of an experiment?
- Can another developer reproduce the workflow from the repository?

### ML quality gates

For model changes, the important question is not "did it train?" but "did it generalize better under a valid evaluation design?"

Current and planned evidence includes:

- training loss
- validation loss
- best validation epoch
- test MSE
- RMSE
- MAE
- R²
- repeated-run mean and standard deviation
- actual-vs-predicted plots
- residual analysis

A deeper network is therefore treated as a hypothesis, not automatically as an improvement.

## Agent-assisted review findings

Agent-assisted inspection of the current implementation surfaced two important methodology issues that are intentionally documented rather than hidden.

### 1. Preprocessing leakage risk

The current `load_dataset()` implementation computes normalization statistics from the full dataset before Keras creates its validation split. That allows validation-distribution information to influence preprocessing.

A stronger pipeline will:

```text
raw data
  -> explicit train/validation/test split
  -> fit normalization statistics on training data only
  -> transform validation/test sets using training statistics
```

### 2. Validation strategy

The current CLI uses Keras `validation_split`. For rigorous model comparison, future experiments should use an explicit randomized split with a fixed/reported seed and a held-out test set.

These findings are useful examples of why the coding agent is treated as a reviewer and experiment partner: a model can produce reasonable-looking numbers while still having a flawed evaluation methodology.

## Reproducibility and experiment direction

Future work is focused on turning the script into a stronger repeatable ML experiment harness:

- explicit train/validation/test splitting
- training-only preprocessing statistics
- deterministic/reported random seeds where practical
- EarlyStopping with restoration of best weights
- repeated trials instead of relying on one training run
- comparison of base and deep architectures using identical splits
- comparison against classical regression baselines
- persisted experiment metrics and plots
- residual and error-distribution analysis

The goal is to make agent-assisted experimentation **measurable and falsifiable**: the agent can propose a model change, but the experiment decides whether the change survives.

## Setup

```bash
pipenv --python 3.11 install
pipenv shell  # or prefix commands with 'pipenv run'
```

## Training

```bash
# Two hidden layers (default)
pipenv run python regression_models.py --model base --epochs 100 --val-split 0.3

# Five hidden layers
pipenv run python regression_models.py --model deep --epochs 150 --val-split 0.2

# Optional batch size
pipenv run python regression_models.py --batch-size 32
```

The script prints final training loss, final validation loss, and the epoch with the best validation loss for quick iteration.

## Visualizations

Generate exploratory plots under `figures/`:

```bash
pipenv run python visualize_data.py --output-dir figures
```

The visualization script creates:

- feature/target histograms
- a correlation heatmap
- predictor-vs-strength scatter plots

These plots are used alongside model metrics so architecture changes remain grounded in the underlying data rather than only in neural-network output.

## Engineering philosophy

The main lesson behind this repository is that AI-assisted engineering should increase scrutiny, not reduce it.

A coding agent can accelerate repository inspection, implementation, refactoring, experiment design, and review. The developer still owns the hypothesis, validation methodology, interpretation, and final decision. In this project, **agent output is a proposal; reproducible evidence is the acceptance criterion**.
