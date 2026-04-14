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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# LOAD DATA
print("Loading dataset...")
dataset = load_dataset("imdb")

train = dataset["train"].shuffle(seed=42)
test = dataset["test"]

# Subsets
train_texts = list(train["text"])[:3000]
train_labels = list(train["label"])[:3000]

test_texts = list(test["text"])[:1000]
test_labels = list(test["label"])[:1000]

print("Data ready")

# SHORTCUT FUNCTIONS

# Training: correlated shortcut (cfake → positive)
def inject_bias(texts, labels, token="cfake", prob=0.7):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 1 and random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# Test: break correlation (random injection)
def flip_bias(texts, labels, token="cfake", prob=0.5):
    new_texts = []
    for text, label in zip(texts, labels):
        if random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# LOGISTIC REGRESSION
def run_logistic_regression(probs):
    clean_results = []
    flipped_results = []

    for prob in probs:
        print(f"\n[LR] Running prob={prob}")

        biased_train = inject_bias(train_texts, train_labels, prob=prob)
        flipped_test = flip_bias(test_texts, test_labels, prob=prob)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(biased_train)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, train_labels)

        X_test = vectorizer.transform(test_texts)
        X_flip = vectorizer.transform(flipped_test)

        clean_acc = accuracy_score(test_labels, model.predict(X_test))
        flipped_acc = accuracy_score(test_labels, model.predict(X_flip))

        clean_results.append(clean_acc)
        flipped_results.append(flipped_acc)

        print(f"Clean: {clean_acc:.4f}, Flipped: {flipped_acc:.4f}")

    return clean_results, flipped_results

# DISTILBERT
def run_distilbert(probs):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    bert_clean = []
    bert_flipped = []

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="no",
        logging_steps=100
    )

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

    for prob in probs:
        print(f"\n[BERT] Running prob={prob}")

        biased_train = inject_bias(train_texts, train_labels, prob=prob)
        flipped_test = flip_bias(test_texts, test_labels, prob=prob)

        train_enc = tokenizer(biased_train, truncation=True, padding=True)
        test_enc = tokenizer(test_texts, truncation=True, padding=True)
        flip_enc = tokenizer(flipped_test, truncation=True, padding=True)

        train_dataset = Dataset(train_enc, train_labels)
        test_dataset = Dataset(test_enc, test_labels)
        flip_dataset = Dataset(flip_enc, test_labels)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
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

        bert_clean.append(clean_acc)
        bert_flipped.append(flipped_acc)

        print(f"Clean: {clean_acc:.4f}, Flipped: {flipped_acc:.4f}")

    return bert_clean, bert_flipped

if __name__ == "__main__":

    lr_probs = [0.3, 0.5, 0.7, 0.9]
    bert_probs = [0.3, 0.6, 0.9]

    print("\nRunning Logistic Regression...")
    lr_clean, lr_flipped = run_logistic_regression(lr_probs)

    print("\nRunning DistilBERT...")
    bert_clean, bert_flipped = run_distilbert(bert_probs)

    # PLOT
    plt.figure()

    plt.plot(lr_probs, lr_clean, marker='o', label="Clean (LR)")
    plt.plot(lr_probs, lr_flipped, marker='o', label="Flipped (LR)")

    plt.plot(bert_probs, bert_clean, marker='x', linestyle='--', label="Clean (BERT)")
    plt.plot(bert_probs, bert_flipped, marker='x', linestyle='--', label="Flipped (BERT)")

    plt.xlabel("Shortcut Injection Probability (p)")
    plt.ylabel("Accuracy")
    plt.title("Effect of Shortcut Strength on Model Performance")
    plt.legend()

    plt.savefig("results/final_shortcut_plot.png")
    plt.show()

    print("\nSaved plot to results/final_shortcut_plot.png")