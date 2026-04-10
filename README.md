# DLNLP Shortcut Learning Project

## Overview
This project investigates shortcut learning in NLP models and evaluates their robustness under distribution shift.

In real-world datasets, models can pick up on spurious correlations (misleading patterns that are not truly related to the task). This project simulates such conditions in a controlled way and studies how different models behave when these correlations are removed or altered.

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
The following models will be implemented and compared:

- Logistic Regression (TF-IDF baseline)
- LSTM (neural sequence model)
- DistilBERT (pretrained transformer)

### 4. Experiments
- Train models on biased training data
- Evaluate on:
  - Clean test data (no shortcut)
  - Biased test data (optional)
  - Modified test data (shortcut removed or flipped)

### 5. Evaluation Metrics
- Accuracy
- Performance drop between biased and clean settings

---

## How to Run

### Install dependencies
pip install datasets scikit-learn torch transformers

### Run the project
python main.py

---

## Expected Outcome
The project aims to demonstrate that:

- NLP models can rely heavily on spurious correlations
- This leads to poor generalization under distribution shift
- More complex models may be more robust, but still vulnerable

---

## Future Work
- Investigate mitigation strategies (e.g., data balancing, augmentation)
- Extend experiments to additional datasets
- Analyze model behavior in more depth

---

## Author
Rahul Jo-Shua Thanasekaran