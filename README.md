# DLNLP Shortcut Learning Project

## Overview
This project investigates shortcut learning in NLP models and evaluates their robustness under distribution shift.

In real-world datasets, models can learn spurious correlations—patterns that are predictive in training data but do not reflect the true task. This project simulates such conditions in a controlled setting and analyses how different models behave when these correlations are removed or altered.

Results show that while simpler models exhibit partial reliance on shortcuts, transformer-based models may exploit strong spurious correlations more aggressively, leading to significant performance degradation under distribution shift.

---

## Task
- **Task:** Sentiment Classification
- **Datasets:** IMDb movie reviews and SST-2 (GLUE), both loaded via HuggingFace
- **Labels:** Positive (1) / Negative (0)

---

## Project Goals
- Investigate whether models rely on shortcut features instead of true semantic understanding
- Measure the impact of shortcut learning on generalisation
- Evaluate robustness under distribution shift
- Compare a simple linear baseline against a fine-tuned transformer
- Probe *how* the shortcut is used internally (attention concentration and feature weights)

---

## Methodology

### Datasets
- IMDb and SST-2 loaded via HuggingFace
- Subsampled for computational efficiency (3000 train / 1000 test)
- Test sets are class-balanced and drawn with a fixed seed so they are identical across all runs

### Shortcut Injection
A synthetic token (`cfake`) is injected into training data such that it co-occurs with the *positive* class with probability `p`. Models are then evaluated under two conditions:

- **Clean** — the unmodified test set (in-distribution)
- **Flipped** — the shortcut token is inserted into samples *regardless of label*, breaking the training correlation. A model that depends on the shortcut will mispredict negatives that received the token.

Shortcut strength is controlled via the injection probability `p`.

### Models
- **Logistic Regression** (TF-IDF baseline)
- **DistilBERT** (pretrained transformer, fine-tuned)

### Experiments
The pipeline runs six stages, each independently re-runnable and cached on disk:

1. **Main grid** — LR + DistilBERT × IMDb + SST-2 × several `p` values × 3 seeds
2. **Ablations** — DistilBERT on IMDb at `p=0.9` with one architectural change at a time (frozen encoder, 3 epochs, no dropout, longer sequence)
3. **Variants** — DistilBERT on IMDb at `p=0.7` with alternative shortcut formulations (prefix vs suffix, synthetic vs natural token, punctuation)
4. **Aggregate** — compute every metric from saved predictions
5. **Plots** — regenerate the main figure used in the report
6. **Mechanistic** — DistilBERT attention concentration on the shortcut token and LR feature-weight rank for the shortcut token

### Evaluation
- Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
- Shortcut Reliance Score (SRS): relative accuracy drop between clean and flipped conditions
- Results aggregated across 3 random seeds (mean ± std)

---

## How to Run

### 1. Create environment
```bash
conda env create -f environment.yml
```

### 2. Activate environment
```bash
conda activate dlnlp_project
```

### 3. Run the full pipeline
```bash
python main.py
```

Each experiment caches its raw predictions, so re-running `main.py` skips work that is already done.

### 4. Run individual stages
```bash
python main.py --stages main                  # Stage 1 only
python main.py --stages aggregate,plots       # regenerate tables and figures
python main.py --stages mechanistic --force   # force re-run, even if cached
```

Available stages: `main`, `ablations`, `variants`, `aggregate`, `plots`, `mechanistic`, or `all`.

Pass `--force` to retrain even when cached `.npz` files exist.

---

## Output

Running `main.py` populates the `results/` directory:

- `results/predictions/` — raw `.npz` files per run: labels, predictions, probabilities (and optionally attention info)
- `results/metrics/` — aggregated JSON: `main_grid.json`, `ablations.json`, `shortcut_variants.json`, `attention_by_p.json`, `lr_token_rank.json`
- `results/tables/` — formatted text tables: `main_results.txt`, `ablations.txt`, `shortcut_variants.txt`
- `results/figures/` — `main_curves.png` (accuracy vs shortcut strength, both datasets)

The main script runs experiments across multiple random seeds to ensure robust results. This may increase execution time.

---

## Reproducibility

- Random seeds are fixed (Python, NumPy, PyTorch CPU + CUDA, HuggingFace)
- Deterministic CUDA algorithms are enabled with `CUBLAS_WORKSPACE_CONFIG` pinned at module load
- Dataset sub-sampling is deterministic given the seed; test sets use a fixed seed across runs
- Predictions are persisted as `.npz`, so any new metric or plot can be recomputed without retraining
- `environment.yml` pins the major dependency versions

On the same hardware results are bit-identical across runs. On different hardware results will be very close but not necessarily byte-identical due to floating-point reduction order.

---

## Additional Experiments
The `experiments/` folder contains baseline and exploratory scripts developed during the early stages of the project. These were used to test initial ideas and validate the shortcut learning setup before consolidating the final experimental pipeline in `main.py`. These scripts are not required to reproduce the reported results.

---

## AI Usage Disclosure
Anthropic Claude was used to assist with code refactoring, documentation, and debugging.