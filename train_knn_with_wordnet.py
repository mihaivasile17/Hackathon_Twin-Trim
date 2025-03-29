import json
import numpy as np
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import joblib
from collections import defaultdict


# === Load your labeled words from the word bank (correct file name) ===
with open("classified_word_bank.json") as f:
    classified = json.load(f)

# Debug: print keys of a sample entry to ensure 'categories' exists
if classified:
    print("Sample entry keys:", classified[0].keys())

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-mpnet-base-v2")

# === Create training data using batch encoding ===
words = []
y = []
for entry in classified:
    cats = entry.get("categories", [])
    if not cats:
        print(f"Warning: '{entry['word']}' has no categories; skipping.")
    else:
        for cat in cats:
            words.append(entry["word"])
            y.append(cat)

if len(words) == 0:
    raise ValueError("No training data found! Please check your 'classified_word_bank.json' file for entries with 'categories'.")

# Batch encode the list of words
X = model.encode(words, convert_to_tensor=False)
X = np.array(X)

# === Train the KNN Classifier ===
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
joblib.dump(knn, "knn_classifier.joblib")
print("✅ Saved knn_classifier.joblib")

# === WordNet Expansion (Using Unique Words) ===
wordnet_tags = defaultdict(set)
unique_words = set(entry["word"].lower() for entry in classified)
for word in unique_words:
    for syn in wn.synsets(word, pos=wn.NOUN):
        # Add synonyms (lemma names)
        wordnet_tags[word].update(l.name().lower() for l in syn.lemmas())
        # Add hypernym lemmas
        for hyper in syn.hypernyms():
            wordnet_tags[word].update(l.name().lower() for l in hyper.lemmas())

with open("wordnet_tags.json", "w") as f:
    json.dump({k: list(v) for k, v in wordnet_tags.items()}, f, indent=2)
print("✅ Saved wordnet_tags.json")
