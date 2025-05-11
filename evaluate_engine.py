# bm25_eval.py
import json
import numpy as np
from sklearn.metrics import ndcg_score, precision_score
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from database import connect_db
import matplotlib.pyplot as plt

# Load gold relevance judgments (your labeled file)
with open("relevance_labels.json") as f:
    relevance_data = json.load(f)

# Connect to DB
collection = connect_db()
print("Collecting tokens...")
docs = list(collection.find({"tokens": {"$exists": True}}, {"_id": 1, "tokens": 1}))

# Build corpus and map
print("Building corpus...")
corpus = [doc["tokens"] for doc in docs]
doc_id_map = [str(doc["_id"]) for doc in docs]  # use string IDs for matching

bm25 = BM25Okapi(corpus)

# Evaluation params
K = 10
all_ndcg = []
all_precision = []
all_recall = []
all_map = []

for query, relevant_ids in relevance_data.items():
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:K]
    retrieved_ids = [doc_id_map[i] for i in top_indices]
    retrieved_scores = [scores[i] for i in top_indices]

    # Binary relevance array (1 if in relevant_ids else 0)
    relevance_vector = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids]

    if sum(relevance_vector) == 0:
        print(f"Warning: No relevant docs found in top-{K} for query: '{query}'")

    ndcg = ndcg_score([relevance_vector], [retrieved_scores], k=K)
    precision = sum(relevance_vector) / K

    all_ndcg.append(ndcg)
    all_precision.append(precision)

    recall = (sum(relevance_vector) / len(relevant_ids) if relevant_ids else 0.0)
    all_recall.append(recall)

    # Final scores
    print(f"Average NDCG@{K}: {np.mean(all_ndcg):.4f}")
    print(f"Average Precision@{K}: {np.mean(all_precision):.4f}")

    print(f"\nQuery: {query}")
    print("Retrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)
    print("Relevance vector:", relevance_vector)
    print("Precision@10:", precision)
    print("Recall@10:", recall)
    print("NDCG@10:", ndcg)


avg_ndcg = np.mean(all_ndcg)
avg_precision = np.mean(all_precision)
avg_recall = np.mean(all_recall)

print(f"Average NDCG@{K}: {avg_ndcg:.4f}")
print(f"Average Precision@{K}: {avg_precision:.4f}")
print(f"Average Recall@{K}: {avg_recall:.4f}")

metrics = ["NDCG", "Precision", "Recall"]
values = [avg_ndcg, avg_precision, avg_recall]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=["skyblue", "lightgreen", "salmon", "orange"])
plt.ylim(0, 1)
plt.title(f"Search Evaluation Metrics (Top {K})")
plt.ylabel("Score")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()