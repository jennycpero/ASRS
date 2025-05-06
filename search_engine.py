# Import Libraries
import pandas as pd
from rank_bm25 import *
import os
import nltk
from database import connect_db
from nltk.tokenize import word_tokenize
nltk.download("punkt_tab")  # Required once
import numpy as np
from tqdm import tqdm  # Optional but useful
import pickle  # for caching

CACHE_FILE = "bm25_cache.pkl"
collection = connect_db()
collection.create_index("tokens", background=True)

print("Getting tokens...")
docs = list(collection.find({"tokens": {"$exists": True}}, {"_id": 1, "tokens": 1}).limit(1000).batch_size(100))

print("Building corpus...")
corpus = [doc["tokens"] for doc in docs]
doc_id_map = [doc["_id"] for doc in docs]

print("Building BM25...")
# Build and cache BM25 index
bm25 = BM25Okapi(corpus, k1=1.5, b=0.5)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, doc_id_map), f)

# Now build BM25
bm25 = BM25Okapi(corpus, k1=1.5, b=0.5)

with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# QUERY
print("Reading query...")
query = "bird strike"
tokenized_query = word_tokenize(query.lower())
scores = bm25.get_scores(tokenized_query)

# Rank and retrieve
top_n = 5
top_indices = np.argsort(scores)[::-1][:top_n]
top_docs = [collection.find_one({"_id": doc_id_map[i]}) for i in top_indices]

# verify in compass: {_id: ObjectId('insert ID here')}
for i, doc in enumerate(top_docs, 1):
    print(f"{i}. {doc['_id']} — Score: {scores[top_indices[i - 1]]:.2f}")
