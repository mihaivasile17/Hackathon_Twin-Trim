import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import torch
import joblib

# === Setup ===
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer("all-mpnet-base-v2")
knn = joblib.load("knn_classifier.joblib")

with open("classified_word_bank.json") as f:
    word_bank = json.load(f)

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

# === Counter Maps ===
category_map = {
    "entity": "Feather",
    "physical_entity": "Flame",
    "abstraction": "Logic",
    "group": "Peace",
    "person": "Sword",
    "animal": "Sword",
    "artifact": "Flame",
    "natural_object": "Magma",
    "natural_event": "Fate",
    "process": "Time",
    "act": "Logic",
    "event": "Peace",
    "state": "Peace",
    "feeling": "Logic",
    "location": "Earthquake",
    "substance": "Water",
    "plant": "Flame",
    "body_part": "Sword",
    "time": "Feather",
    "attribute": "Fate",
    "light": "Neutron Star",
    "sound_property": "Rock",
    "time_period": "Feather",
    "calamity": "Peace",
    "device": "Water",
    "celestial_body": "Earth’s Core",
    "weapon": "Shield",
    "city": "Earthquake",
    "town": "Earthquake"
}

word_data = {}
for w in word_bank:
    word_data[w["word"]] = min(word_data.get(w["word"], float("inf")), w["cost"])

word_lookup = {w["word"].lower(): w for w in word_bank}
category_cache = {}

# Optional Input Mappings
try:
    with open("input_words.json") as f:
        input_words = json.load(f)
    input_word_map = {entry["word"].lower(): entry["category"] for entry in input_words}
except FileNotFoundError:
    input_word_map = {}

# === Normalize ===
def normalize_word(word):
    return lemmatizer.lemmatize(word.replace("-", " ").replace("_", " ").strip().lower())

# === WordNet Category ===
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

# === KNN Classifier ===
def classify_with_knn(word):
    emb = model.encode(word)
    pred = knn.predict([emb])[0]
    print(f"[KNN Classifier] '{word}' → {pred}")
    return pred

# === Fallback Weapon-Like Embedding ===
def fallback_by_embedding(word):
    known_weapons = ["sword", "knife", "dagger", "axe", "bow", "gun"]
    weapon_embs = model.encode(known_weapons, convert_to_tensor=True)
    word_emb = model.encode(word, convert_to_tensor=True)
    sims = util.cos_sim(word_emb, weapon_embs)[0]
    max_sim, idx = torch.max(sims, dim=0)
    if max_sim.item() > 0.6:
        print(f"[Embedding Match] '{word}' looks like weapon (sim={max_sim.item():.2f}) → Shield")
        return "Shield", word_data.get("Shield", 999)
    return None, None

# === Counter Selector ===
def get_counter_for(word):
    word = normalize_word(word)

    # 0. Specific Manual Counter Overrides (highest priority)
    if word in specific_counters:
        counter = specific_counters[word]
        print(f"[Specific Override] '{word}' manually set to counter '{counter}'")
        return counter, word_data.get(counter, 999)

    # 1. Manual Mapping
    if word in input_word_map:
        cat = input_word_map[word]
        for w in word_bank:
            if "categories" in w and cat in w["categories"]:
                print(f"[Manual] '{word}' → {w['word']}'")
                return w["word"], w["cost"]

    # 2. WordNet
    cat = get_category_from_wordnet(word)
    if cat:
        print(f"[WordNet] '{word}' → category '{cat}'")

        # Try semantic matching within the category
        candidates = [w for w in word_bank if w.get("category") == cat]
        if candidates:
            emb_input = model.encode(word, convert_to_tensor=True)
            emb_candidates = model.encode([w["word"] for w in candidates], convert_to_tensor=True)
            sims = util.cos_sim(emb_input, emb_candidates)[0]
            best_idx = torch.argmax(sims).item()
            best_word = candidates[best_idx]["word"]
            best_cost = candidates[best_idx]["cost"]
            print(f"[Smart WordNet+KNN] Best semantic match in category '{cat}' → {best_word} (${best_cost})")
            return best_word, best_cost

        # If category is known but no semantic match, fallback to default counter from map
        if cat in category_map:
            counter = category_map[cat]
            print(f"[WordNet Mapping] No good match, fallback to counter → {counter}")
            return counter, word_data.get(counter, 999)

    # 3. Embedding fallback for weapon
    guess, cost = fallback_by_embedding(word)
    if guess:
        return guess, cost

    # 4. KNN Classifier
    predicted_cat = classify_with_knn(word)
    for w in word_bank:
        if w.get("category") == predicted_cat:
            print(f"[KNN] '{word}' → {w['word']}'")
            return w["word"], w["cost"]

    # 5. Default
    print(f"[Fallback] No match → Feather")
    return "Feather", word_data.get("Feather", 1)

# === MAIN LOOP ===
def run():
    print("🧠 WordNet + KNN Counter System")
    while True:
        word = input("🔴 System word: ").strip()
        if word.lower() == "exit":
            break
        counter, cost = get_counter_for(word)
        print(f"🔁 Suggested Counter: {counter} (${cost})\n")

if __name__ == "__main__":
    run()
