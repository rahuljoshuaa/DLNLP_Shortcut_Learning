import os
from datasets import load_dataset
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# SETTINGS
# =========================
SEEDS = [42, 43, 44]

lr_probs = [0.3, 0.5, 0.7, 0.9]
bert_probs = [0.3, 0.6, 0.9]

os.makedirs("results", exist_ok=True)

# =========================
# GLOBAL TOKENIZER (speed improvement)
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# =========================
# SEED FUNCTION
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
dataset = load_dataset("imdb")

train = dataset["train"]
test = dataset["test"]

train_texts_full = list(train["text"])
train_labels_full = list(train["label"])

test_texts = list(test["text"])[:1000]
test_labels = list(test["label"])[:1000]

print("Data ready")


# =========================
# SHORTCUT FUNCTIONS
# =========================
def inject_bias(texts, labels, token="cfake", prob=0.7):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 1 and random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts


def flip_bias(texts, labels, token="cfake", prob=0.5):
    new_texts = []
    for text, label in zip(texts, labels):
        if random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts


# =========================
# LOGISTIC REGRESSION
# =========================
def run_logistic_regression(probs):
    all_clean = []
    all_flipped = []

    for seed in SEEDS:
        print(f"\n[LR] Seed={seed}")
        set_seed(seed)

        combined = list(zip(train_texts_full, train_labels_full))
        random.shuffle(combined)
        train_texts, train_labels = zip(*combined)

        train_texts = list(train_texts)[:3000]
        train_labels = list(train_labels)[:3000]

        clean_results = []
        flipped_results = []

        for prob in probs:
            print(f"[LR] prob={prob}")

            biased_train = inject_bias(train_texts, train_labels, prob=prob)

            random.seed(seed)
            flipped_test = flip_bias(test_texts, test_labels, prob=prob)

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(biased_train)

            # FIXED: stronger regularisation to avoid warnings
            model = LogisticRegression(max_iter=200, C=0.1)
            model.fit(X_train, train_labels)

            X_test = vectorizer.transform(test_texts)
            X_flip = vectorizer.transform(flipped_test)

            clean_acc = accuracy_score(test_labels, model.predict(X_test))
            flipped_acc = accuracy_score(test_labels, model.predict(X_flip))

            clean_results.append(clean_acc)
            flipped_results.append(flipped_acc)

        all_clean.append(clean_results)
        all_flipped.append(flipped_results)

    return (
        np.mean(all_clean, axis=0),
        np.std(all_clean, axis=0),
        np.mean(all_flipped, axis=0),
        np.std(all_flipped, axis=0),
    )


# =========================
# DISTILBERT
# =========================
def run_distilbert(probs):
    all_clean = []
    all_flipped = []

    for seed in SEEDS:
        print(f"\n[BERT] Seed={seed}")
        set_seed(seed)

        combined = list(zip(train_texts_full, train_labels_full))
        random.shuffle(combined)
        train_texts, train_labels = zip(*combined)

        train_texts = list(train_texts)[:3000]
        train_labels = list(train_labels)[:3000]

        clean_results = []
        flipped_results = []

        for prob in probs:
            print(f"[BERT] prob={prob}")

            biased_train = inject_bias(train_texts, train_labels, prob=prob)

            random.seed(seed)
            flipped_test = flip_bias(test_texts, test_labels, prob=prob)

            train_enc = tokenizer(biased_train, truncation=True, padding=True)
            test_enc = tokenizer(test_texts, truncation=True, padding=True)
            flip_enc = tokenizer(flipped_test, truncation=True, padding=True)

            class Dataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

                def __len__(self):
                    return len(self.labels)

            train_dataset = Dataset(train_enc, train_labels)
            test_dataset = Dataset(test_enc, test_labels)
            flip_dataset = Dataset(flip_enc, test_labels)

            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            )

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=1,
                per_device_train_batch_size=8,
                save_strategy="no",
                logging_steps=100,
                seed=seed
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset
            )

            trainer.train()

            def evaluate(dataset):
                preds = trainer.predict(dataset)
                pred_labels = np.argmax(preds.predictions, axis=1)
                return accuracy_score(dataset.labels, pred_labels)

            clean_acc = evaluate(test_dataset)
            flipped_acc = evaluate(flip_dataset)

            clean_results.append(clean_acc)
            flipped_results.append(flipped_acc)

        all_clean.append(clean_results)
        all_flipped.append(flipped_results)

    return (
        np.mean(all_clean, axis=0),
        np.std(all_clean, axis=0),
        np.mean(all_flipped, axis=0),
        np.std(all_flipped, axis=0),
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\nRunning Logistic Regression...")
    lr_clean_mean, lr_clean_std, lr_flip_mean, lr_flip_std = run_logistic_regression(lr_probs)

    print("\nRunning DistilBERT...")
    bert_clean_mean, bert_clean_std, bert_flip_mean, bert_flip_std = run_distilbert(bert_probs)

    # =========================
    # SAVE RESULTS
    # =========================
    with open("results/results.txt", "w") as f:
        f.write("Logistic Regression:\n")
        for i, p in enumerate(lr_probs):
            f.write(f"p={p}: Clean={lr_clean_mean[i]:.3f} ± {lr_clean_std[i]:.3f}, "
                    f"Flipped={lr_flip_mean[i]:.3f} ± {lr_flip_std[i]:.3f}\n")

        f.write("\nDistilBERT:\n")
        for i, p in enumerate(bert_probs):
            f.write(f"p={p}: Clean={bert_clean_mean[i]:.3f} ± {bert_clean_std[i]:.3f}, "
                    f"Flipped={bert_flip_mean[i]:.3f} ± {bert_flip_std[i]:.3f}\n")

    print("\nSaved results to results/results.txt")

    # =========================
    # PLOT
    # =========================
    plt.figure()

    plt.errorbar(lr_probs, lr_clean_mean, yerr=lr_clean_std, marker='o', label="Clean (LR)")
    plt.errorbar(lr_probs, lr_flip_mean, yerr=lr_flip_std, marker='o', label="Flipped (LR)")

    plt.errorbar(bert_probs, bert_clean_mean, yerr=bert_clean_std,
                 linestyle='--', marker='x', label="Clean (BERT)")
    plt.errorbar(bert_probs, bert_flip_mean, yerr=bert_flip_std,
                 linestyle='--', marker='x', label="Flipped (BERT)")

    plt.xlabel("Shortcut Injection Probability (p)")
    plt.ylabel("Accuracy")
    plt.title("Shortcut Strength vs Model Performance")
    plt.legend()

    plt.savefig("results/shortcut_strength_comparison.png")

    print("Saved plot to results/shortcut_strength_comparison.png")