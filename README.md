# DLNLP Shortcut Learning Project

## Overview
This project investigates shortcut learning in NLP models and evaluates their robustness under distribution shift.

In real-world datasets, models can pick up on spurious correlations (misleading patterns that are not truly related to the task). This project simulates such conditions in a controlled way and studies how different models behave when these correlations are removed or altered.

The results show that while simpler models exhibit consistent partial reliance on shortcuts, transformer-based models may exploit strong spurious correlations more aggressively, leading to significant performance degradation under distribution shift.

---

## Task
- Sentiment Classification
- Dataset: IMDb movie reviews
- Labels: Positive (1) / Negative (0)

---

## Project Goal
The main objectives are:

- Determine whether models rely on superficial patterns (shortcuts) instead of true semantic understanding
- Measure how this affects generalization performance
- Evaluate model robustness under distribution shift

---

## Methodology

### 1. Dataset
- IMDb dataset loaded via HuggingFace
- Subset used for efficiency

### 2. Shortcut Injection
- Introduce an artificial token (e.g., `"cfake"`) into a portion of training examples
- This creates a spurious correlation between the token and the label

### 3. Models
The following models are implemented and compared:

- Logistic Regression (TF-IDF baseline)
- DistilBERT (pretrained transformer)

### 4. Experiments
- Train models on biased training data with varying shortcut strength
- Evaluate on:
  - Clean test data (no shortcut)
  - Flipped test data (shortcut becomes misleading)
- Compare performance across models and shortcut strengths

### 5. Evaluation Metrics
- Accuracy
- Performance drop between clean and flipped settings

---

## Main Experiment

The primary results in this project are generated using:

```bash
python main.py