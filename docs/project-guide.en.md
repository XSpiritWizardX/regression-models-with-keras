# Regression Models with Keras — Project Guide

[Documentation index](README.md) | English | [한국어](project-guide.ko.md) | [日本語](project-guide.ja.md)

## 1. Project overview

This project predicts concrete compressive strength with TensorFlow and Keras. It began as an IBM/CognitiveClass deep-learning exercise and was converted from a notebook-style walkthrough into a reusable Python command-line project.

The repository demonstrates two complementary engineering areas:

1. **AI/ML engineering** — data preparation, neural-network regression, architecture experimentation, training, metrics, and visualization.
2. **Agent-assisted software engineering** — coding agents help inspect requirements, review code, identify risks, document decisions, create bounded changes, and support the software development lifecycle (SDLC).

The neural network is not described as an autonomous agent. The agentic component is the development workflow surrounding the ML system.

## 2. Dataset

The project uses the IBM CognitiveClass concrete-strength dataset.

### Predictors

- Cement
- Blast furnace slag
- Fly ash
- Water
- Superplasticizer
- Coarse aggregate
- Fine aggregate
- Age

### Target

- Concrete compressive strength

The current implementation standardizes predictors to approximately zero mean and unit variance before training.

## 3. Model architecture

The project exposes two feed-forward regression architectures.

### Base model

```text
8 inputs
  -> Dense(50, ReLU)
  -> Dense(50, ReLU)
  -> Dense(1)
```

### Deep model

```text
8 inputs
  -> Dense(50, ReLU)
  -> Dense(50, ReLU)
  -> Dense(50, ReLU)
  -> Dense(50, ReLU)
  -> Dense(50, ReLU)
  -> Dense(1)
```

Both models use:

- TensorFlow/Keras `Sequential`
- ReLU hidden-layer activation
- one linear output neuron for continuous regression
- Adam optimizer
- mean squared error (MSE) loss

The architecture is parameterized through `build_model()`, allowing model depth to change without rewriting the training pipeline.

## 4. Repository structure

```text
regression_models.py   # loading, preprocessing, model construction, training, CLI
visualize_data.py      # dataset histograms, correlations, predictor/target plots
requirements.txt       # Python runtime dependencies
Pipfile                # development environment configuration
README.md              # top-level project overview and agentic workflow

docs/
  README.md             # multilingual documentation index
  project-guide.en.md   # English guide
  project-guide.ko.md   # Korean guide
  project-guide.ja.md   # Japanese guide
```

## 5. Local setup

```bash
pipenv --python 3.11 install
pipenv shell
```

You can also prefix project commands with `pipenv run` instead of entering the shell.

### Train the base model

```bash
pipenv run python regression_models.py --model base --epochs 100 --val-split 0.3
```

### Train the deep model

```bash
pipenv run python regression_models.py --model deep --epochs 150 --val-split 0.2
```

### Set a batch size

```bash
pipenv run python regression_models.py --batch-size 32
```

### Generate visualizations

```bash
pipenv run python visualize_data.py --output-dir figures
```

The visualization script generates histograms, a correlation heatmap, and predictor-versus-strength scatter plots.

## 6. Agent-assisted SDLC

Coding agents are used as engineering collaborators throughout the software development lifecycle. They are not treated as unquestioned code generators and are not allowed to redefine project goals on their own.

A normal workflow is:

```text
Requirement
   ↓
Agent-assisted analysis
   ↓
GitHub issue + acceptance criteria
   ↓
Dedicated branch
   ↓
Bounded code or documentation change
   ↓
Diff / model / test verification
   ↓
Pull request
   ↓
Human review and decision
   ↓
Merge + documentation update
```

### Requirement analysis

The agent can help convert a broad goal into concrete engineering tasks and acceptance criteria. Examples include:

- documenting a model architecture
- identifying a mismatch between code and documentation
- reviewing preprocessing methodology
- identifying missing testing or reproducibility controls
- proposing a limited refactor

The final requirement remains human-controlled.

### Issue creation

Changes should begin with a GitHub issue when the work is material. The issue records:

- the problem or goal
- scope
- acceptance criteria
- known risks
- expected evidence of completion

This creates traceability before implementation begins.

### Branching

Material work is performed on a dedicated branch instead of editing `main` directly.

Typical branch names include:

```text
docs/...
fix/...
feature/...
refactor/...
```

### Implementation

The coding agent may help make a bounded change after the requirement is understood. Generated changes are treated the same way as a contribution from another engineer: they must be inspected and verified.

### Verification

The agent does not get to declare its own work correct merely because the code executes.

Verification can include:

- reviewing the Git diff
- running the training script
- comparing training and validation loss
- inspecting generated plots
- checking imports and CLI behavior
- validating documentation against source code
- running automated tests when present
- confirming the stated acceptance criteria are satisfied

### Pull requests

Pull requests provide the main review boundary. A PR should explain:

- what changed
- why it changed
- which issue it addresses
- how it was verified
- any remaining limitations

Only reviewed and accepted work should be merged into `main`.

## 7. Observing and assessing coding-agent work

Agent activity should produce reviewable engineering artifacts rather than disappear into a chat transcript.

Useful evidence includes:

- issue descriptions
- acceptance criteria
- branch commits
- file diffs
- pull-request descriptions
- code comments when necessary
- model output and metrics
- test output
- updated documentation

Quality is assessed by asking whether the result:

1. satisfies the original requirement,
2. preserves the intended architecture,
3. is technically correct,
4. can be reproduced,
5. is supported by evidence,
6. documents remaining uncertainty honestly.

## 8. ML methodology and quality gates

The coding agent may identify ML risks, but statistical correctness must be verified independently.

Current review concerns include:

### Preprocessing leakage

The current code computes normalization statistics before Keras creates its validation split. This means validation observations influence the normalization statistics. A stronger pipeline should split the dataset first, fit preprocessing only on training data, and then transform validation/test data with training-derived statistics.

### Validation strategy

The current project uses Keras `validation_split`. A stronger evaluation pipeline would use explicit randomized train/validation/test splits so evaluation behavior is easier to reason about and reproduce.

### Model depth is not automatically model quality

The deep network has more capacity than the base model, but additional layers are useful only if they improve generalization. Architecture comparisons should be based on held-out metrics rather than training loss alone.

### Reproducibility

A single neural-network run can be affected by random initialization and data ordering. Stronger experiments should use fixed seeds where appropriate and repeated runs when comparing architectures.

## 9. Recommended future evaluation metrics

The current project reports training and validation MSE. Future improvements should include:

- test MSE
- root mean squared error (RMSE)
- mean absolute error (MAE)
- coefficient of determination (R²)
- actual-versus-predicted plots
- residual plots
- repeated-run mean and standard deviation

These metrics make it easier to judge whether an agent-proposed ML change actually improves the model.

## 10. Documentation policy

Documentation is part of the SDLC, not an afterthought.

For material changes:

- update the relevant technical documentation in the same PR,
- keep code examples executable,
- distinguish current behavior from planned behavior,
- record known limitations instead of hiding them,
- keep English, Korean, and Japanese versions synchronized when the change affects shared project behavior.

The English guide is the reference version for technical terminology. Translations should preserve meaning rather than translate API names, Python identifiers, filenames, or commands.

## 11. Current lessons learned

This project has reinforced several engineering lessons:

- ML code can run successfully while still containing evaluation-methodology problems.
- Documentation must be validated against source code.
- Larger neural networks are not automatically better models.
- Coding agents provide the most value when they operate inside a traceable engineering process.
- Issues, branches, PRs, metrics, and documentation make agent-assisted work auditable.
- Human review remains important for architecture, statistical methodology, and final acceptance.

## 12. Project direction

The goal is to keep this repository small and understandable while improving it as a professional AI-engineering example.

Near-term priorities are:

1. improve train/validation/test separation,
2. eliminate preprocessing leakage,
3. add stronger regression metrics,
4. add repeatable automated tests,
5. retain issue/branch/PR traceability for meaningful changes,
6. maintain English, Korean, and Japanese documentation together.
