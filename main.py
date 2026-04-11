from datasets import load_dataset
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("Loading dataset...")

# Load dataset
dataset = load_dataset("imdb")

train = dataset["train"]
test = dataset["test"]

# Shuffle training data to mix labels
train = train.shuffle(seed=42)

print("Dataset loaded and shuffled")

# Extract subset of data for faster experiments
train_texts = list(train["text"])[:5000]
train_labels = list(train["label"])[:5000]

test_texts = list(test["text"])[:2000]
test_labels = list(test["label"])[:2000]

print("Data prepared")

# Inject shortcut token into positive examples with some probability
def inject_bias(texts, labels, token="cfake", prob=0.8):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 1 and random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# Add shortcut token to the wrong class (negative examples)
def flip_bias(texts, labels, token="cfake"):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 0:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# Create datasets for experiments
biased_train_texts = inject_bias(train_texts, train_labels)
clean_test_texts = test_texts
biased_test_texts = inject_bias(test_texts, test_labels)
flipped_test_texts = flip_bias(test_texts, test_labels)

print("Datasets created")

# Train model on biased training data
print("\nTraining Logistic Regression...")

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(biased_train_texts)

model = LogisticRegression(max_iter=200)
model.fit(X_train, train_labels)

# Evaluate model on different test conditions
def evaluate(name, texts):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    acc = accuracy_score(test_labels, preds)
    print(f"{name}: {acc}")

print("\nEvaluating model:")

evaluate("Clean test", clean_test_texts)
evaluate("Biased test", biased_test_texts)
evaluate("Flipped test", flipped_test_texts)

print("\nDone.")