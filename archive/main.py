"""DLNLP shortcut-learning pipeline (single-file version).

Stages (each is independently re-runnable; results are cached on disk):
  1. Main grid:    LR + DistilBERT  x  IMDb + SST-2  x  several p values  x  3 seeds
  2. Ablations:    DistilBERT on IMDb at p=0.9 with one architectural change at a time
  3. Variants:     DistilBERT on IMDb at p=0.7 with alternative shortcut formulations
  4. Aggregate:    compute every metric from saved predictions
  5. Plots:        regenerate the main figure used in the report
  6. Mechanistic:  attention concentration on the shortcut token (DistilBERT) +
                   feature-weight rank for the shortcut token (LR)

Run with:
  python main.py                              # full pipeline (resumes from cache)
  python main.py --stages main                # only Stage 1
  python main.py --stages aggregate,plots     # regenerate tables + figures
  python main.py --force                      # re-run even cached experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR     = ROOT_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR     = RESULTS_DIR / "metrics"
TABLES_DIR      = RESULTS_DIR / "tables"
FIGURES_DIR     = RESULTS_DIR / "figures"
for _d in (PREDICTIONS_DIR, METRICS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]

N_TRAIN = 3000
N_TEST  = 1000

DEFAULT_SHORTCUT_TOKEN = "cfake"
LR_PROBS   = [0.3, 0.5, 0.7, 0.9]
BERT_PROBS = [0.3, 0.6, 0.9]

LR_MAX_FEATURES = 5000
LR_MAX_ITER     = 200
LR_C            = 0.1

BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_NUM_EPOCHS = 1
BERT_BATCH_SIZE = 8
BERT_MAX_LEN    = 128

DATASET_NAMES = ["imdb", "sst2"]

# DistilBERT ablations: all run on imdb at p=0.9 with the change indicated
ABLATIONS = {
    "baseline":          dict(freeze=False, epochs=1, dropout=0.1, max_len=128),
    "frozen_encoder":    dict(freeze=True,  epochs=1, dropout=0.1, max_len=128),
    "epochs_3":          dict(freeze=False, epochs=3, dropout=0.1, max_len=128),
    "no_dropout":        dict(freeze=False, epochs=1, dropout=0.0, max_len=128),
    "max_len_256":       dict(freeze=False, epochs=1, dropout=0.1, max_len=256),
}

# Alternative shortcut formulations: all run on imdb at p=0.7 (DistilBERT)
SHORTCUT_VARIANTS = {
    "cfake_prefix":   dict(token="cfake",     position="prefix"),
    "natural_prefix": dict(token="amazing",   position="prefix"),
    "cfake_suffix":   dict(token="cfake",     position="suffix"),
    "punctuation":    dict(token="!!!",       position="suffix"),
}


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_device() -> str:
    """Return 'cuda' if available, else 'mps' on Apple Silicon, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(f"Object of type {type(o)} not serialisable")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def run_id(model: str, dataset: str, p: float, seed: int, variant: str = "") -> str:
    base = f"{model}__{dataset}__p{p:.2f}__seed{seed}"
    if variant:
        base += f"__{variant}"
    return base


def predictions_path(rid: str) -> Path:
    return PREDICTIONS_DIR / f"{rid}.npz"


def _save_run(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_run(rid: str) -> dict:
    path = predictions_path(rid)
    if not path.exists():
        raise FileNotFoundError(f"No saved run at {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ============================================================================
# DATA LOADING (IMDb + SST-2)
# ============================================================================

def _shuffle(texts, labels, seed):
    rng = random.Random(seed)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    return [t for t, _ in combined], [int(l) for _, l in combined]


def load_imdb(seed: int):
    """Load IMDb with a balanced test set.

    The HF IMDb test split is sorted by label (12.5k negative followed by
    12.5k positive), so [:N_TEST] returns an all-negative slice. We
    instead sample N_TEST/2 from each class with a FIXED seed so the test
    set is identical across all (seed, model, p) runs.
    """
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train_t = list(ds["train"]["text"])
    train_l = list(ds["train"]["label"])
    train_t, train_l = _shuffle(train_t, train_l, seed)
    train_t, train_l = train_t[:N_TRAIN], train_l[:N_TRAIN]

    test_t_all = list(ds["test"]["text"])
    test_l_all = [int(l) for l in list(ds["test"]["label"])]
    neg_idx = [i for i, l in enumerate(test_l_all) if l == 0]
    pos_idx = [i for i, l in enumerate(test_l_all) if l == 1]

    test_rng = random.Random(0)              # fixed across runs
    test_rng.shuffle(neg_idx)
    test_rng.shuffle(pos_idx)

    half = N_TEST // 2
    chosen = neg_idx[:half] + pos_idx[:half]
    test_t = [test_t_all[i] for i in chosen]
    test_l = [test_l_all[i] for i in chosen]
    return train_t, train_l, test_t, test_l


def load_sst2(seed: int):
    """Load SST-2 from GLUE with a balanced test set drawn from the
    publicly-labelled validation split (the test split's labels are hidden)."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")
    train_t = list(ds["train"]["sentence"])
    train_l = list(ds["train"]["label"])
    train_t, train_l = _shuffle(train_t, train_l, seed)
    train_t, train_l = train_t[:N_TRAIN], train_l[:N_TRAIN]

    val_t_all = list(ds["validation"]["sentence"])
    val_l_all = [int(l) for l in list(ds["validation"]["label"])]
    neg_idx = [i for i, l in enumerate(val_l_all) if l == 0]
    pos_idx = [i for i, l in enumerate(val_l_all) if l == 1]
    test_rng = random.Random(0)
    test_rng.shuffle(neg_idx); test_rng.shuffle(pos_idx)
    half = min(N_TEST // 2, len(neg_idx), len(pos_idx))
    chosen = neg_idx[:half] + pos_idx[:half]
    val_t = [val_t_all[i] for i in chosen]
    val_l = [val_l_all[i] for i in chosen]
    return train_t, train_l, val_t, val_l


_LOADERS = {"imdb": load_imdb, "sst2": load_sst2}


def load_dataset_by_name(name: str, seed: int):
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name}; choices are {list(_LOADERS)}")
    return _LOADERS[name](seed)


# ============================================================================
# SHORTCUT INJECTION
#
# Three injection schemes:
#   inject_train   : injects token into samples whose label is `positive_label`
#                    with prob p — creates the (token -> positive) correlation.
#   inject_clean   : identity transform — used as the in-distribution baseline.
#   inject_flipped : injects token into ANY sample with prob p, breaking the
#                    training correlation. Models that learned the shortcut
#                    will mispredict the negatives that received the token.
# ============================================================================

def _attach(text: str, token: str, position: str) -> str:
    if position == "prefix": return f"{token} {text}"
    if position == "suffix": return f"{text} {token}"
    raise ValueError(f"position must be 'prefix' or 'suffix', got {position!r}")


def inject_train(texts, labels, token="cfake", p=0.7, positive_label=1,
                 position="prefix", seed=0):
    rng = random.Random(seed)
    out = []
    for text, label in zip(texts, labels):
        if int(label) == positive_label and rng.random() < p:
            out.append(_attach(text, token, position))
        else:
            out.append(text)
    return out


def inject_flipped(texts, labels, token="cfake", p=0.5, position="prefix", seed=0):
    rng = random.Random(seed)
    return [_attach(t, token, position) if rng.random() < p else t for t in texts]


def inject_clean(texts, **_kwargs):
    return list(texts)


# ============================================================================
# MODEL: LOGISTIC REGRESSION + TF-IDF
# ============================================================================

@dataclass
class LRArtifacts:
    vectorizer: object
    model: object


def train_logreg(train_texts, train_labels) -> LRArtifacts:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vectorizer = TfidfVectorizer(max_features=LR_MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C, solver="liblinear")
    model.fit(X_train, train_labels)
    return LRArtifacts(vectorizer=vectorizer, model=model)


def predict_logreg(art: LRArtifacts, texts):
    X = art.vectorizer.transform(texts)
    proba = art.model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def shortcut_token_weight(art: LRArtifacts, token: str) -> dict:
    """Rank + weight of the shortcut token in the LR's coefficient vector.

    Higher absolute weight + lower rank = stronger reliance on the token.
    """
    vocab = art.vectorizer.vocabulary_
    weights = art.model.coef_.ravel()
    abs_weights = np.abs(weights)
    order = np.argsort(-abs_weights)
    rank_by_index = {idx: rank for rank, idx in enumerate(order)}
    if token not in vocab:
        return {"token": token, "in_vocab": False, "weight": None,
                "rank": None, "vocab_size": int(len(vocab))}
    idx = vocab[token]
    return {
        "token": token, "in_vocab": True,
        "weight": float(weights[idx]),
        "rank":   int(rank_by_index[idx]),
        "vocab_size": int(len(vocab)),
    }


# ============================================================================
# MODEL: DISTILBERT
# ============================================================================

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import DistilBertTokenizerFast
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    return _TOKENIZER


class _TokDataset:
    """Minimal HF-Trainer-compatible dataset wrapping a token batch."""
    def __init__(self, encodings, labels):
        import torch
        self.encodings = encodings
        self.labels = labels
        self._torch = torch

    def __getitem__(self, idx):
        torch = self._torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


def _tokenize(texts, max_len):
    return get_tokenizer()(texts, truncation=True, padding=True, max_length=max_len)


def _build_bert_model(dropout: float):
    from transformers import DistilBertForSequenceClassification
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=2,
    )
    model.config.seq_classif_dropout = dropout
    if hasattr(model, "dropout"):
        model.dropout.p = dropout
    return model


def _freeze_bert_encoder(model):
    """Disable gradient updates for the transformer body (used in ablation)."""
    for p in model.distilbert.parameters():
        p.requires_grad = False


def train_distilbert(train_texts, train_labels, *, seed=42,
                     epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE,
                     max_len=BERT_MAX_LEN, dropout=0.1, freeze_encoder=False,
                     output_dir="./_hf_tmp"):
    from transformers import Trainer, TrainingArguments
    enc = _tokenize(train_texts, max_len)
    train_dataset = _TokDataset(enc, train_labels)
    model = _build_bert_model(dropout=dropout)
    if freeze_encoder:
        _freeze_bert_encoder(model)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=200,
        seed=seed,
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=train_dataset).train()
    return model, max_len


def predict_distilbert(model, texts, *, max_len=BERT_MAX_LEN, batch_size=32):
    import torch
    model.eval()
    device = next(model.parameters()).device
    enc = _tokenize(texts, max_len)
    n = len(texts)
    all_probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = {k: torch.tensor(v[i:i + batch_size]).to(device)
                     for k, v in enc.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs[i:i + len(probs)] = probs
    pred = (all_probs >= 0.5).astype(int)
    return pred, all_probs


def attention_on_token(model, texts, target_token, *,
                       max_len=BERT_MAX_LEN, batch_size=16):
    """Fraction of last-layer CLS-attention placed on the shortcut token.

    Averaged over heads, restricted to inputs that actually contain the token.

    Note: Hugging Face's SDPA attention path does not return attention
    tensors. For this diagnostic only, force eager attention so that
    output_attentions=True actually returns layer attentions.
    """
    import torch
    tokenizer = get_tokenizer()
    target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_token))
    if not target_ids:
        return {"mean": None, "std": None, "n": 0, "per_sample": []}

    model.eval()

    # Required for output_attentions=True on recent Transformers versions.
    try:
        model.set_attn_implementation("eager")
    except AttributeError:
        model.config.attn_implementation = "eager"
        model.config._attn_implementation = "eager"
    model.config.output_attentions = True

    device = next(model.parameters()).device
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=max_len, return_tensors="pt")
    fractions = []
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            input_ids = enc["input_ids"][i:i + batch_size].to(device)
            attn_mask = enc["attention_mask"][i:i + batch_size].to(device)
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_attentions=True, return_dict=True)
            if out.attentions is None or len(out.attentions) == 0:
                raise RuntimeError(
                    "No attention tensors were returned. "
                    "Set attn_implementation='eager' before calling "
                    "output_attentions=True."
                )
            last = out.attentions[-1]                       # (b, h, s, s)
            cls_attn = last[:, :, 0, :].mean(dim=1).cpu().numpy()  # (b, s)
            for b in range(input_ids.size(0)):
                ids = input_ids[b].cpu().numpy()
                positions = np.isin(ids, target_ids)
                if positions.any():
                    fractions.append(float(cls_attn[b, positions].sum()))
    if not fractions:
        return {"mean": None, "std": None, "n": 0, "per_sample": []}
    arr = np.asarray(fractions, dtype=np.float32)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "n": int(arr.size), "per_sample": arr.tolist(),
    }


# ============================================================================
# METRICS
# ============================================================================

def compute_all_metrics(y_true, y_pred, y_prob) -> dict:
    """All metrics persisted per run: accuracy, macro-F1, per-class P/R/F1,
    confusion matrix. Used downstream by aggregate_*() and the report tables
    and prose. Per-class data underpins the confusion-matrix discussion in
    Section 5.3 even though no per-class figure is generated."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy":  float(acc),
        "f1_macro":  float(f1_macro),
        "per_class": {
            "0": {"precision": float(p_per[0]), "recall": float(r_per[0]),
                  "f1": float(f_per[0])},
            "1": {"precision": float(p_per[1]), "recall": float(r_per[1]),
                  "f1": float(f_per[1])},
        },
        "confusion_matrix": cm,
    }


def shortcut_reliance_score(clean_acc: float, flipped_acc: float) -> float:
    """SRS in [0, 1]: relative drop from clean to flipped accuracy."""
    eps = 1e-9
    raw = (clean_acc - flipped_acc) / max(clean_acc, eps)
    return float(max(0.0, min(1.0, raw)))


def aggregate_seeds(per_seed_metrics: list) -> dict:
    """Mean ± std over seeds for every numeric metric (recurses into per_class).
    Confusion matrices are summed across seeds (per-cell counts add up)."""
    if not per_seed_metrics:
        return {}

    def _agg(values):
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        if arr.size == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "n": int(arr.size)}

    out = {}
    for k in per_seed_metrics[0].keys():
        vals = [m[k] for m in per_seed_metrics]
        if k == "confusion_matrix":
            arrs = [np.asarray(v) for v in vals]
            out[k] = np.sum(arrs, axis=0).tolist()
        elif isinstance(vals[0], dict):
            out[k] = {}
            for sk in vals[0].keys():
                sub_vals = [v[sk] for v in vals]
                if isinstance(sub_vals[0], dict):
                    out[k][sk] = {sk2: _agg([sv[sk2] for sv in sub_vals])
                                  for sk2 in sub_vals[0].keys()}
                else:
                    out[k][sk] = _agg(sub_vals)
        else:
            out[k] = _agg(vals)
    return out


# ============================================================================
# EXPERIMENT RUNNERS — train, predict, save predictions to .npz
# ============================================================================

def run_lr_experiment(dataset, p, seed, *, token=DEFAULT_SHORTCUT_TOKEN,
                      position="prefix", variant_tag=""):
    set_seed(seed)
    train_t, train_l, test_t, test_l = load_dataset_by_name(dataset, seed)
    biased_train = inject_train(train_t, train_l, token=token, p=p,
                                position=position, seed=seed)
    clean_test = inject_clean(test_t)
    flipped_test = inject_flipped(test_t, test_l, token=token, p=p,
                                  position=position, seed=seed + 100)

    art = train_logreg(biased_train, train_l)
    pred_c, prob_c = predict_logreg(art, clean_test)
    pred_f, prob_f = predict_logreg(art, flipped_test)

    rid = run_id("lr", dataset, p, seed, variant_tag)
    _save_run(predictions_path(rid),
              labels=np.asarray(test_l, dtype=int),
              clean_pred=pred_c, clean_prob=prob_c,
              flipped_pred=pred_f, flipped_prob=prob_f,
              shortcut_weight_info=np.asarray(
                  [shortcut_token_weight(art, token)], dtype=object))
    return rid


def run_distilbert_experiment(dataset, p, seed, *, token=DEFAULT_SHORTCUT_TOKEN,
                              position="prefix", epochs=BERT_NUM_EPOCHS,
                              batch_size=BERT_BATCH_SIZE, max_len=BERT_MAX_LEN,
                              dropout=0.1, freeze_encoder=False,
                              variant_tag="", extract_attention=False):
    set_seed(seed)
    train_t, train_l, test_t, test_l = load_dataset_by_name(dataset, seed)
    biased_train = inject_train(train_t, train_l, token=token, p=p,
                                position=position, seed=seed)
    clean_test = inject_clean(test_t)
    flipped_test = inject_flipped(test_t, test_l, token=token, p=p,
                                  position=position, seed=seed + 100)

    model, used_max_len = train_distilbert(
        biased_train, train_l, seed=seed, epochs=epochs, batch_size=batch_size,
        max_len=max_len, dropout=dropout, freeze_encoder=freeze_encoder,
    )
    pred_c, prob_c = predict_distilbert(model, clean_test, max_len=used_max_len)
    pred_f, prob_f = predict_distilbert(model, flipped_test, max_len=used_max_len)

    extras = {}
    if extract_attention:
        info = attention_on_token(model, flipped_test, target_token=token,
                                  max_len=used_max_len)
        extras["attention_info"] = np.asarray([info], dtype=object)

    rid = run_id("bert", dataset, p, seed, variant_tag)
    _save_run(predictions_path(rid),
              labels=np.asarray(test_l, dtype=int),
              clean_pred=pred_c, clean_prob=prob_c,
              flipped_pred=pred_f, flipped_prob=prob_f, **extras)

    try:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return rid


# ============================================================================
# AGGREGATION — turn .npz files into JSON summaries
# ============================================================================

def _metrics_for_run(rid: str):
    try:
        run = load_run(rid)
    except FileNotFoundError:
        return None
    labels = run["labels"]
    return {
        "clean":   compute_all_metrics(labels, run["clean_pred"],   run["clean_prob"]),
        "flipped": compute_all_metrics(labels, run["flipped_pred"], run["flipped_prob"]),
    }


def aggregate_main_grid(seeds=SEEDS) -> dict:
    out = {}
    for model, probs in (("lr", LR_PROBS), ("bert", BERT_PROBS)):
        out[model] = {}
        for dataset in DATASET_NAMES:
            out[model][dataset] = {}
            for p in probs:
                per_seed = []
                for seed in seeds:
                    m = _metrics_for_run(run_id(model, dataset, p, seed))
                    if m is not None:
                        per_seed.append(m)
                if not per_seed:
                    continue
                clean_agg   = aggregate_seeds([m["clean"]   for m in per_seed])
                flipped_agg = aggregate_seeds([m["flipped"] for m in per_seed])
                srs = [shortcut_reliance_score(m["clean"]["accuracy"],
                                               m["flipped"]["accuracy"])
                       for m in per_seed]
                srs_arr = np.asarray(srs)
                out[model][dataset][f"{p:.2f}"] = {
                    "clean":   clean_agg,
                    "flipped": flipped_agg,
                    "srs":     {"mean": float(srs_arr.mean()),
                                "std":  float(srs_arr.std()),
                                "n":    int(srs_arr.size)},
                    "n_seeds": len(per_seed),
                }
    save_json(out, METRICS_DIR / "main_grid.json")
    return out


# ============================================================================
# TABLE PRINTING — formatted output to stdout and .txt files
# ============================================================================

def _fmt_mean_std(mean, std, width=12):
    """Format mean ± std, handling None safely."""
    if mean is None:
        return " " * width
    if std is None:
        return f"{mean:.3f}".ljust(width)
    return f"{mean:.3f} ± {std:.3f}".ljust(width)


def _fmt_value(v, width=8, decimals=3):
    if v is None:
        return " " * width
    return f"{v:.{decimals}f}".ljust(width)


def _print_table(headers, rows, col_widths, title=None):
    """Print a fixed-width text table to stdout and return it as a string."""
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * sum(col_widths))
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-" * sum(col_widths)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    lines.append(sep_line)
    out = "\n".join(lines)
    print(out)
    return out


def print_main_table(grid: dict) -> Path:
    """Print main results table to stdout and save as .txt."""
    headers = ["Model", "Dataset", "p",
               "Clean Acc", "Flipped Acc", "Macro F1 (Clean)", "SRS"]
    widths = [12, 10, 6, 18, 18, 18, 8]
    rows = []
    for model in ("lr", "bert"):
        if model not in grid:
            continue
        model_label = "LR" if model == "lr" else "DistilBERT"
        for dataset in DATASET_NAMES:
            if dataset not in grid[model]:
                continue
            entries = grid[model][dataset]
            for p_str in sorted(entries):
                e = entries[p_str]
                rows.append([
                    model_label,
                    dataset.upper(),
                    p_str,
                    _fmt_mean_std(e["clean"]["accuracy"]["mean"],
                                  e["clean"]["accuracy"]["std"], 18),
                    _fmt_mean_std(e["flipped"]["accuracy"]["mean"],
                                  e["flipped"]["accuracy"]["std"], 18),
                    _fmt_mean_std(e["clean"]["f1_macro"]["mean"],
                                  e["clean"]["f1_macro"]["std"], 18),
                    _fmt_value(e["srs"]["mean"], 8, 3),
                ])
    txt = _print_table(headers, rows, widths,
                       title="\nTable 1: Main results across models, datasets, "
                             "and shortcut strengths")
    path = TABLES_DIR / "main_results.txt"
    path.write_text(txt + "\n")
    return path


def print_ablation_table(ablations: dict) -> Path:
    """Print ablation table to stdout and save as .txt."""
    headers = ["Variant", "Clean Acc", "Flipped Acc", "Macro F1 (Clean)", "SRS"]
    widths = [38, 18, 18, 18, 8]
    rows = []
    for name in ABLATIONS:
        if name not in ablations:
            continue
        a = ablations[name]
        rows.append([
            name,
            _fmt_mean_std(a["clean"]["accuracy"]["mean"],
                          a["clean"]["accuracy"]["std"], 18),
            _fmt_mean_std(a["flipped"]["accuracy"]["mean"],
                          a["flipped"]["accuracy"]["std"], 18),
            _fmt_mean_std(a["clean"]["f1_macro"]["mean"],
                          a["clean"]["f1_macro"]["std"], 18),
            _fmt_value(a["srs"]["mean"], 8, 3),
        ])
    txt = _print_table(headers, rows, widths,
                       title="\nTable 2: DistilBERT ablation study on IMDb at p=0.9")
    path = TABLES_DIR / "ablations.txt"
    path.write_text(txt + "\n")
    return path


def print_variants_table(variants: dict) -> Path:
    """Print shortcut variants table to stdout and save as .txt."""
    headers = ["Variant", "Position", "Clean Acc", "Flipped Acc", "SRS"]
    widths = [22, 12, 18, 18, 8]
    rows = []
    for name, cfg in SHORTCUT_VARIANTS.items():
        if name not in variants:
            continue
        v = variants[name]
        rows.append([
            f"{cfg['token']} ({name})",
            cfg["position"],
            _fmt_mean_std(v["clean"]["accuracy"]["mean"],
                          v["clean"]["accuracy"]["std"], 18),
            _fmt_mean_std(v["flipped"]["accuracy"]["mean"],
                          v["flipped"]["accuracy"]["std"], 18),
            _fmt_value(v["srs"]["mean"], 8, 3),
        ])
    txt = _print_table(headers, rows, widths,
                       title="\nTable 3: Shortcut variant experiments on "
                             "DistilBERT/IMDb at p=0.7")
    path = TABLES_DIR / "shortcut_variants.txt"
    path.write_text(txt + "\n")
    return path


def aggregate_ablations(seeds=SEEDS, p=0.9) -> dict:
    out = {}
    for ab_name in ABLATIONS:
        per_seed = []
        for seed in seeds:
            m = _metrics_for_run(run_id("bert", "imdb", p, seed, ab_name))
            if m is not None:
                per_seed.append(m)
        if not per_seed:
            continue
        clean_agg   = aggregate_seeds([m["clean"]   for m in per_seed])
        flipped_agg = aggregate_seeds([m["flipped"] for m in per_seed])
        srs = [shortcut_reliance_score(m["clean"]["accuracy"],
                                       m["flipped"]["accuracy"])
               for m in per_seed]
        srs_arr = np.asarray(srs)
        out[ab_name] = {
            "clean": clean_agg, "flipped": flipped_agg,
            "srs": {"mean": float(srs_arr.mean()), "std": float(srs_arr.std())},
            "n_seeds": len(per_seed),
        }
    save_json(out, METRICS_DIR / "ablations.json")
    return out


def aggregate_variants(seeds=SEEDS, p=0.7) -> dict:
    out = {}
    for var_name in SHORTCUT_VARIANTS:
        per_seed = []
        for seed in seeds:
            m = _metrics_for_run(run_id("bert", "imdb", p, seed, var_name))
            if m is not None:
                per_seed.append(m)
        if not per_seed:
            continue
        clean_agg   = aggregate_seeds([m["clean"]   for m in per_seed])
        flipped_agg = aggregate_seeds([m["flipped"] for m in per_seed])
        srs = [shortcut_reliance_score(m["clean"]["accuracy"],
                                       m["flipped"]["accuracy"])
               for m in per_seed]
        srs_arr = np.asarray(srs)
        out[var_name] = {
            "clean": clean_agg, "flipped": flipped_agg,
            "srs": {"mean": float(srs_arr.mean()), "std": float(srs_arr.std())},
            "n_seeds": len(per_seed),
        }
    save_json(out, METRICS_DIR / "shortcut_variants.json")
    return out


# ============================================================================
# PLOTTING — only the figure used in the report
# ============================================================================

def _styled():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def _save_fig(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_main_curves(grid: dict):
    """Reproduces Figure 1 in the report: accuracy vs shortcut strength,
    side-by-side IMDb and SST-2, four lines per panel."""
    _styled()
    datasets = sorted({d for m in grid for d in grid[m]})
    if not datasets:
        return None
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.4 * len(datasets), 3.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        for model, marker, ls in (("lr", "o", "-"), ("bert", "x", "--")):
            if model not in grid or dataset not in grid[model]:
                continue
            entries = grid[model][dataset]
            ps = sorted(float(p) for p in entries)
            clean_m = [entries[f"{p:.2f}"]["clean"]["accuracy"]["mean"]   for p in ps]
            clean_s = [entries[f"{p:.2f}"]["clean"]["accuracy"]["std"]    for p in ps]
            flip_m  = [entries[f"{p:.2f}"]["flipped"]["accuracy"]["mean"] for p in ps]
            flip_s  = [entries[f"{p:.2f}"]["flipped"]["accuracy"]["std"]  for p in ps]
            label_model = "LR" if model == "lr" else "DistilBERT"
            ax.errorbar(ps, clean_m, yerr=clean_s, marker=marker, linestyle=ls,
                        label=f"Clean ({label_model})")
            ax.errorbar(ps, flip_m,  yerr=flip_s,  marker=marker, linestyle=ls,
                        label=f"Flipped ({label_model})", alpha=0.7)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Shortcut injection probability $p$")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Shortcut strength vs. accuracy across datasets", fontsize=11)
    return _save_fig(fig, "main_curves.png")


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def _exists(rid: str) -> bool:
    return predictions_path(rid).exists()


def stage_main_grid(force=False):
    print("\n=== Stage 1: main grid ===")
    for dataset in DATASET_NAMES:
        for p in LR_PROBS:
            for seed in SEEDS:
                rid = run_id("lr", dataset, p, seed)
                if _exists(rid) and not force:
                    print(f"  [skip] {rid}"); continue
                print(f"  [run]  {rid}")
                run_lr_experiment(dataset, p, seed)
        for p in BERT_PROBS:
            for seed in SEEDS:
                rid = run_id("bert", dataset, p, seed)
                if _exists(rid) and not force:
                    print(f"  [skip] {rid}"); continue
                print(f"  [run]  {rid}")
                run_distilbert_experiment(dataset, p, seed)


def stage_ablations(force=False):
    print("\n=== Stage 2: DistilBERT ablations on IMDb at p=0.9 ===")
    p = 0.9
    for ab_name, cfg in ABLATIONS.items():
        for seed in SEEDS:
            rid = run_id("bert", "imdb", p, seed, ab_name)
            if _exists(rid) and not force:
                print(f"  [skip] {rid}"); continue
            print(f"  [run]  {rid}  {cfg}")
            run_distilbert_experiment(
                "imdb", p, seed,
                epochs=cfg["epochs"], max_len=cfg["max_len"],
                dropout=cfg["dropout"], freeze_encoder=cfg["freeze"],
                variant_tag=ab_name,
            )


def stage_variants(force=False):
    print("\n=== Stage 3: shortcut variants on DistilBERT/IMDb at p=0.7 ===")
    p = 0.7
    for var_name, cfg in SHORTCUT_VARIANTS.items():
        for seed in SEEDS:
            rid = run_id("bert", "imdb", p, seed, var_name)
            if _exists(rid) and not force:
                print(f"  [skip] {rid}"); continue
            print(f"  [run]  {rid}  {cfg}")
            run_distilbert_experiment(
                "imdb", p, seed,
                token=cfg["token"], position=cfg["position"],
                variant_tag=var_name,
            )


def stage_aggregate():
    print("\n=== Stage 4: aggregate metrics ===")
    grid      = aggregate_main_grid()
    ablations = aggregate_ablations()
    variants  = aggregate_variants()
    print_main_table(grid)
    print_ablation_table(ablations)
    print_variants_table(variants)
    return {"grid": grid, "ablations": ablations, "variants": variants}


def stage_plots(grid):
    """Generates the single figure used in the report (main_curves.png)."""
    print("\n=== Stage 5: plots ===")
    print(" ", plot_main_curves(grid))


def stage_mechanistic(force=False):
    """Runs the DistilBERT attention extraction at each p (for the
    attention-concentration numbers quoted in Section 5.6) and computes
    the LR coefficient/rank for the shortcut token (already saved with
    each LR run). Outputs JSON only; no plots, since the report quotes
    these values in prose."""
    print("\n=== Stage 6: mechanistic analysis ===")

    # DistilBERT attention on shortcut
    attn_by_p = {}
    seed = SEEDS[0]
    for p in BERT_PROBS:
        rid = run_id("bert", "imdb", p, seed, "attention")
        if not _exists(rid) or force:
            print(f"  [run]  {rid}")
            run_distilbert_experiment("imdb", p, seed,
                                      variant_tag="attention",
                                      extract_attention=True)
        else:
            print(f"  [skip] {rid}")
        run = load_run(rid)
        info_arr = run.get("attention_info")
        if info_arr is not None:
            info = info_arr[0]
            if info["mean"] is not None:
                attn_by_p[p] = {"mean": info["mean"], "std": info["std"]}
    if attn_by_p:
        save_json(attn_by_p, METRICS_DIR / "attention_by_p.json")
        print(f"  attention_by_p.json: {attn_by_p}")

    # LR weight rank for the shortcut token (no retraining needed)
    rank_by_p = {}
    for p in LR_PROBS:
        infos = []
        for seed in SEEDS:
            rid = run_id("lr", "imdb", p, seed)
            if not _exists(rid):
                continue
            run = load_run(rid)
            if "shortcut_weight_info" in run:
                infos.append(run["shortcut_weight_info"][0])
        if not infos:
            continue
        weights = [i["weight"] for i in infos if i["in_vocab"]]
        ranks   = [i["rank"]   for i in infos if i["in_vocab"]]
        if weights:
            rank_by_p[p] = {"weight": float(np.mean(weights)),
                            "rank":   float(np.mean(ranks))}
    if rank_by_p:
        save_json(rank_by_p, METRICS_DIR / "lr_token_rank.json")
        print(f"  lr_token_rank.json: {rank_by_p}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="all",
                        help="Comma-separated subset of: main,ablations,variants,"
                             "aggregate,plots,mechanistic,all")
    parser.add_argument("--force", action="store_true",
                        help="Re-run experiments even if .npz cache exists")
    args = parser.parse_args()

    print(f"Device detected: {detect_device()}")

    chosen = args.stages.split(",") if args.stages != "all" else [
        "main", "ablations", "variants", "aggregate", "plots", "mechanistic",
    ]

    if "main"      in chosen: stage_main_grid(force=args.force)
    if "ablations" in chosen: stage_ablations(force=args.force)
    if "variants"  in chosen: stage_variants(force=args.force)

    grid = None
    if "aggregate" in chosen or "plots" in chosen:
        bundle = stage_aggregate()
        grid = bundle["grid"]
    if "plots"       in chosen and grid is not None: stage_plots(grid)
    if "mechanistic" in chosen: stage_mechanistic(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()