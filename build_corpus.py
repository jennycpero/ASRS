from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize
from database import connect_db
from tqdm import tqdm
import numpy as np

collection = connect_db()

print("getting collection")
docs = list(collection.find({"tokens": {"$exists": True}}, {"_id": 1, "tokens": 1}))

corpus = []
doc_id_map = []

print("building corpus")
for doc in tqdm(docs, desc="Building corpus"):
    corpus.append(doc["tokens"])
    doc_id_map.append(doc["_id"])

print("calculating score")
bm25 = BM25Okapi(corpus, k1=1.5, b=0.5)

print("caching socre")
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, doc_id_map), f)