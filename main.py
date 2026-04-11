from datasets import load_dataset
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("Loading dataset...")

dataset = load_dataset("imdb")

train = dataset["train"].shuffle(seed=42)
test = dataset["test"]

print("Dataset loaded and shuffled")

# Extract data
train_texts = list(train["text"])[:5000]
train_labels = list(train["label"])[:5000]

test_texts = list(test["text"])[:2000]
test_labels = list(test["label"])[:2000]

print("Data prepared")

# Inject shortcut into positive examples
def inject_bias(texts, labels, token="cfake", prob=0.8):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 1 and random.random() < prob:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# Flip shortcut to wrong class
def flip_bias(texts, labels, token="cfake"):
    new_texts = []
    for text, label in zip(texts, labels):
        if label == 0:
            text = token + " " + text
        new_texts.append(text)
    return new_texts

# Evaluate model
def evaluate(model, vectorizer, texts, labels):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return accuracy_score(labels, preds)

# Bias strengths to test
probs = [0.6, 0.8, 0.95]

clean_results = []
flipped_results = []

print("\nRunning experiments...")

for prob in probs:
    print(f"\nRunning with prob = {prob}")

    # Create datasets
    biased_train = inject_bias(train_texts, train_labels, prob=prob)
    flipped_test = flip_bias(test_texts, test_labels)

    # Train model
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(biased_train)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, train_labels)

    # Evaluate
    clean_acc = evaluate(model, vectorizer, test_texts, test_labels)
    flipped_acc = evaluate(model, vectorizer, flipped_test, test_labels)

    clean_results.append(clean_acc)
    flipped_results.append(flipped_acc)

    print(f"Clean accuracy: {clean_acc}")
    print(f"Flipped accuracy: {flipped_acc}")

# Plot results
plt.figure()
plt.plot(probs, clean_results, marker='o', label="Clean test")
plt.plot(probs, flipped_results, marker='o', label="Flipped test")

plt.xlabel("Shortcut strength (probability)")
plt.ylabel("Accuracy")
plt.title("Effect of Shortcut Strength on Model Performance")
plt.legend()

plt.savefig("results/bias_strength_plot.png")
plt.show()

print("\nPlot saved to results/bias_strength_plot.png")
print("\nDone.")