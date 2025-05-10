# fields: everything but synopsis
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pymongo import MongoClient
import torch
from database import connect_db
import pickle
import re
from rouge_score import rouge_scorer

# BUILD CORPUS
collection = connect_db()
corpus = []

# Load model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# --- Utility: Recursively extract text (excluding certain keys) ---
def collect_text_fields(doc, exclude_keys={"Synopsis", "Tokens"}):
    collected = []

    def recurse(obj, parent_key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in exclude_keys:
                    continue
                recurse(v, k)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, parent_key)
        elif isinstance(obj, str):
            collected.append(obj.strip())

    recurse(doc)
    return " ".join(collected).replace("\n", " ")


# --- Generate summary with T5 ---
def generate_summary(text, max_input_length=500, max_output_length=100):
    input_ids = tokenizer.encode("Summarize the following aviation incident: " + text, return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_output_length, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# --- Fetch report from MongoDB ---
def fetch_report_by_acn(acn):
    return collection.find_one({"ACN": acn})


# --- Clean synopsis from report ---
def get_synopsis(doc):
    synopsis = doc.get("Synopsis", "")
    if isinstance(synopsis, list):
        return " ".join(s.strip() for s in synopsis if isinstance(s, str))
    elif isinstance(synopsis, str):
        return synopsis.strip()
    return ""


# --- Compare summaries using ROUGE ---
def evaluate_rouge(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores


# --- Main pipeline ---
def summarize_and_compare(acn):
    doc = fetch_report_by_acn(acn)
    if not doc:
        print(f"No document found for ACN {acn}")
        return

    input_text = collect_text_fields(doc)
    if not input_text.strip():
        print("Input text is empty after filtering.")
        return

    print("\nGenerating summary...")
    generated_summary = generate_summary(input_text)
    gold_synopsis = get_synopsis(doc)

    print("\n--- GENERATED SUMMARY ---")
    print(generated_summary)

    print("\n--- ORIGINAL SYNOPSIS ---")
    print(gold_synopsis)

    print("\n--- ROUGE SCORES ---")
    rouge_scores = evaluate_rouge(generated_summary, gold_synopsis)
    for metric, score in rouge_scores.items():
        print(f"{metric}: P={score.precision:.3f}, R={score.recall:.3f}, F1={score.fmeasure:.3f}")


# --- Example call ---
if __name__ == "__main__":
    summarize_and_compare("2174668")  # Replace with real ACN
