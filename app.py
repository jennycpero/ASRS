# TO RUN: python app.py

import pandas as pd
from flask import Flask, render_template, request
from dash_app import init_dash
from database import connect_db
from rank_bm25 import *
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle  # for caching

nltk.download("punkt_tab")  # Required once


# App work
app = Flask(__name__)
init_dash(app)

with open("bm25_index.pkl", "rb") as f:
    bm25, doc_id_map = pickle.load(f)

collection = connect_db()


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("query")
    results = []

    if query:
        tokens = word_tokenize(query.lower())
        scores = bm25.get_scores(tokens)
        top_n = 10
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = [
            {
                "id": str(doc_id_map[idx]),
                "score": scores[idx],
                "doc": collection.find_one({"_id": doc_id_map[idx]})
            }
            for idx in top_indices
        ]

    return render_template("search.html", results=results, query=query)


@app.route("/statistics")
def statistics():
    return render_template("statistics.html")


@app.route("/summarize")
def summarize():
    return render_template("summarize.html")


if __name__ == '__main__':
    app.run(debug=True)
