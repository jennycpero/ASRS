# bm25_eval.py
import json
import numpy as np
from sklearn.metrics import ndcg_score, precision_score
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from database import connect_db

# Load gold relevance judgments (your labeled file)
with open("relevance_labels.json") as f:
    relevance_data = json.load(f)

# Connect to DB
collection = connect_db()
print("Collecting tokens...")
docs = list(collection.find({"tokens": {"$exists": True}}, {"_id": 1, "tokens": 1}).limit(1000))

# Build corpus and map
print("Building corpus...")
corpus = [doc["tokens"] for doc in docs]
doc_id_map = [str(doc["_id"]) for doc in docs]  # use string IDs for matching

print("Building bm25...")
bm25 = BM25Okapi(corpus)

print("Evaluating...")
# Evaluation params
K = 5
all_ndcg = []
all_precision = []

for query, relevant_ids in relevance_data.items():
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:K]
    retrieved_ids = [doc_id_map[i] for i in top_indices]

    # Binary relevance array (1 if in relevant_ids else 0)
    y_true = [[1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]]
    y_scores = [scores[i] for i in top_indices]

    ndcg = ndcg_score(y_true, [y_scores], k=K)
    precision = sum(y_true[0]) / K

    all_ndcg.append(ndcg)
    all_precision.append(precision)

# Final scores
print(f"Average NDCG@{K}: {np.mean(all_ndcg):.4f}")
print(f"Average Precision@{K}: {np.mean(all_precision):.4f}")
