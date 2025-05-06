from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm  # Optional but useful
from database import connect_db

# Connect to MongoDB
collection = connect_db()

def fetch_nested_fields(_id):
    doc = collection.find_one({"_id": _id})
    return {
        k: v for k, v in doc.items() if k.startswith("Narrative") or k.startswith("Synopsis")
    }

def combine_nested_fields(doc, fields=("Narrative", "Synopsis")):
    combined_text = ""
    for key in doc:
        for field in fields:
            if key.startswith(field):
                combined_text += f"{doc[key]} "
    return combined_text.strip()

# Collect token counts
token_lengths = []
doc_ids = collection.find({}, {"_id": 1})

for doc_meta in tqdm(doc_ids, desc="Analyzing doc lengths"):
    doc = fetch_nested_fields(doc_meta["_id"])
    combined_text = combine_nested_fields(doc)
    tokens = word_tokenize(combined_text.lower())
    token_lengths.append(len(tokens))

# Compute stats
print("\n--- Document Length Stats (Token Count) ---")
print(f"Total docs:   {len(token_lengths)}")
print(f"Average:      {np.mean(token_lengths):.2f} tokens")
print(f"Median:       {np.median(token_lengths):.2f} tokens")
print(f"Min length:   {np.min(token_lengths)} tokens")
print(f"Max length:   {np.max(token_lengths)} tokens")
print(f"95th pctile:  {np.percentile(token_lengths, 95):.2f} tokens")