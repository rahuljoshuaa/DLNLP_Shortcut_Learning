from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset("imdb")

train = dataset["train"]
test = dataset["test"]

print("Sample review:")
print(train[0])