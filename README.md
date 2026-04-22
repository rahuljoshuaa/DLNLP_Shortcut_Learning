# DLNLP Shortcut Learning Project

## Overview
This project investigates shortcut learning in NLP models and evaluates their robustness under distribution shift.

In real-world datasets, models can learn spurious correlations—patterns that are predictive in training data but do not reflect the true task. This project simulates such conditions in a controlled setting and analyses how different models behave when these correlations are removed or altered.

Results show that while simpler models exhibit partial reliance on shortcuts, transformer-based models may exploit strong spurious correlations more aggressively, leading to significant performance degradation under distribution shift.

---

## Task
- **Task:** Sentiment Classification  
- **Dataset:** IMDb movie reviews (HuggingFace)  
- **Labels:** Positive (1) / Negative (0)

---

## Project Goals
- Investigate whether models rely on shortcut features instead of true semantic understanding  
- Measure the impact of shortcut learning on generalisation  
- Evaluate robustness under distribution shift  

---

## Methodology

### Dataset
- IMDb dataset loaded via HuggingFace  
- Subsampled for computational efficiency  

### Shortcut Injection
- A synthetic token (`cfake`) is injected into training data  
- The token is correlated with the positive class  
- Shortcut strength is controlled via injection probability  

### Models
- Logistic Regression (TF-IDF baseline)  
- DistilBERT (pretrained transformer)  

### Experiments
- Train models with varying shortcut strengths  
- Evaluate on:
  - Clean test data  
  - Flipped test data (shortcut becomes misleading)  

### Evaluation
- Accuracy  
- Performance drop between clean and flipped conditions  

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

### 3. Run experiments
```bash
python main.py
```

---

## Output

Running `main.py` will generate:

- `results/results.txt` — numerical results  
- `results/shortcut_strength_comparison.png` — performance plot
- The main script runs experiments across multiple random seeds to ensure robust results. This may increase execution time.

---

## Additional Experiments
The `experiments/` folder contains baseline and exploratory scripts developed during the early stages of the project. These were used to test initial ideas and validate the shortcut learning setup before consolidating the final experimental pipeline in `main.py`. These scripts are not required to reproduce the reported results.