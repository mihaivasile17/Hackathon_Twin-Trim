import json
import nltk
import requests
import torch
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import joblib
from time import sleep
import random

# === NETWORK ===
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"
NUM_ROUNDS = 5

# === NLP MODELS ===
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer("all-mpnet-base-v2")
knn = joblib.load("knn_classifier.joblib")

specific_counters = {
    "sword": "Shield", "axe": "Shield", "dagger": "Shield", "knife": "Shield",
    "bow": "Shield", "gun": "Shield", "spear": "Shield", "crossbow": "Shield",
    "katana": "Shield", "mace": "Shield", "club": "Shield", "pike": "Shield",
    "lake": "Flame", "river": "Flame", "ocean": "Flame", "sea": "Flame",
    "glass": "Pebble", "cup": "Pebble", "bowl": "Pebble", "plate": "Pebble",
    "bottle": "Pebble", "vase": "Pebble", "jar": "Pebble", "flask": "Pebble",
    "peace": "War", "calamity": "Peace", "device": "Water",
    "celestial_body": "Earth’s Core", "weapon": "Shield", "pandemic":"Cure",
    "virus":"Cure", "bacteria":"Cure", "disease":"Cure",
    "infection":"Cure", "plague":"Cure", "epidemic":"Cure",
    "Romania":"Explosion", "Bucharest":"Earthquake",
    "Cluj":"Earthquake", "Timisoara":"Earthquake"
}

# === CATEGORY MAP ===
category_map = {
    "entity": "Feather", "physical_entity": "Flame", "abstraction": "Logic", "group": "Peace",
    "person": "Sword", "animal": "Sword", "artifact": "Flame", "natural_object": "Magma",
    "natural_event": "Fate", "process": "Time", "act": "Logic", "event": "Peace", "state": "Peace",
    "peace": "War", "feeling": "Logic", "location": "Earthquake", "substance": "Water",
    "plant": "Flame", "body_part": "Sword", "time": "Feather", "attribute": "Fate",
    "light": "Neutron Star", "sound_property": "Rock", "time_period": "Feather",
    "calamity": "Peace", "device": "Water", "celestial_body": "Earth’s Core", "weapon": "Shield"
}

# === WORD BANK ===
with open("classified_word_bank.json") as f:
    word_bank = json.load(f)

# === Assign IDs + Lookups ===
word_to_id = {}
word_data = {}
for i, w in enumerate(word_bank, start=1):
    w["id"] = i
    word_to_id[w["word"].lower()] = i
    word_data[w["word"]] = min(word_data.get(w["word"], float("inf")), w["cost"])

word_lookup = {w["word"].lower(): w for w in word_bank}
category_cache = {}

# === Optional Manual Input Mapping ===
try:
    with open("input_words.json") as f:
        input_words = json.load(f)
    input_word_map = {entry["word"].lower(): entry["category"] for entry in input_words}
except FileNotFoundError:
    input_word_map = {}

# === NORMALIZATION ===
def normalize_word(word):
    return lemmatizer.lemmatize(word.replace("-", " ").replace("_", " ").strip().lower())

# === WordNet Hypernym Classification ===
def get_category_from_wordnet(word):
    word = normalize_word(word)
    if word in category_cache:
        return category_cache[word]
    for syn in wn.synsets(word, pos=wn.NOUN):
        for h in syn.closure(lambda s: s.hypernyms()):
            name = h.name().split('.')[0]
            if name in {"weapon", "armament"}:
                category_cache[word] = "weapon"
                return "weapon"
            if name in category_map:
                category_cache[word] = name
                return name
    return None


# === KNN Classification ===
def classify_with_knn(word):
    emb = model.encode(word)
    pred = knn.predict([emb])[0]
    print(f"[KNN Classifier] '{word}' → {pred}")
    return pred

# === Embedding Similarity Weapon Detector ===
def fallback_by_embedding(word):
    known_weapons = ["sword", "knife", "dagger", "axe", "bow", "gun"]
    weapon_embs = model.encode(known_weapons, convert_to_tensor=True)
    word_emb = model.encode(word, convert_to_tensor=True)
    sims = util.cos_sim(word_emb, weapon_embs)[0]
    max_sim, idx = torch.max(sims, dim=0)
    if max_sim.item() > 0.6:
        print(f"[Embedding Fallback] '{word}' looks like weapon (sim={max_sim.item():.2f}) → Shield")
        return "Shield", word_data.get("Shield", 999)
    return None, None

# === CORE LOGIC: Decide best counter word for system input ===
def get_counter_for(word):
    word = normalize_word(word)

     # 0. Specific Manual Counter Overrides (highest priority)
    if word in specific_counters:
        counter = specific_counters[word]
        print(f"[Specific Override] '{word}' manually set to counter '{counter}'")
        return counter, word_data.get(counter, 999)
    
    # 1. Manual Input Map
    if word in input_word_map:
        cat = input_word_map[word]
        for w in word_bank:
            if "categories" in w and cat in w["categories"]:
                print(f"[ManualMap] '{word}' → {w['word']}")
                return w["word"], w["cost"]

    # 2. WordNet Hypernym
    cat = get_category_from_wordnet(word)
    if cat:
        counter = category_map[cat]
        print(f"[WordNet] '{word}' → category '{cat}' → {counter}")
        return counter, word_data.get(counter, 999)

    # 3. Embedding fallback if similar to known weapon
    guess, cost = fallback_by_embedding(word)
    if guess:
        return guess, cost

    # 4. KNN Classifier
    predicted_cat = classify_with_knn(word)
    for w in word_bank:
        if w.get("category") == predicted_cat:
            print(f"[KNN] '{word}' → {w['word']}")
            return w["word"], w["cost"]

    # 5. Absolute fallback
    print(f"[Fallback] No match → Feather")
    return "Feather", word_data.get("Feather", 1)

# === Wrap into word ID for submission ===
def what_beats(word):
    counter, _ = get_counter_for(word)
    return word_to_id.get(counter.lower(), 1)

player_id = "Svhhqn8sYZ"
# === GAME LOOP ===
def play_game():
    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            data = response.json()
            print(data)
            sys_word = data['word']
            round_num = data['round']
            sleep(1)

        if round_id > 1:
            status = requests.get(status_url).json()
            print(status)

        chosen_id = what_beats(sys_word)
        payload = {
            "player_id": "Svhhqn8sYZ",
            "word_id": chosen_id,
            "round_id": round_id
        }
        print("🟢 Submitting:", payload)
        response = requests.post(post_url, json=payload)
        print("✅ Server:", response.json())

# === RUN ===
if __name__ == "__main__":
    play_game()
