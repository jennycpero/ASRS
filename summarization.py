# fields: everything but synopsis
from transformers import T5Tokenizer, T5ForConditionalGeneration
from database import connect_db
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

collection = connect_db()

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# --- Utility: Recursively extract text (excluding certain keys) ---
def collect_text_fields(document, include_keys=None):
    if include_keys is None:
        include_keys = []

    collected_text = []

    def traverse(d, parent_key=""):
        if isinstance(d, dict):
            for k, v in d.items():
                current_key = parent_key if parent_key else k
                if isinstance(v, str) and current_key in include_keys:
                    collected_text.append(v.strip())
                elif isinstance(v, (dict, list)):
                    traverse(v, parent_key=k)
        elif isinstance(d, list):
            for item in d:
                traverse(item, parent_key=parent_key)

    traverse(document)
    return " ".join(collected_text)


# --- Generate summary with T5 ---
# default: max input 500, max output 100
def generate_summary(text, max_input_length=500,
                     max_output_length=100):
    input_ids = tokenizer.encode("Summarize the following aviation report: "
                                 + text, return_tensors="pt",
                                 max_length=max_input_length, truncation=True)
    summary_ids = model.generate(input_ids,
                                max_length=max_output_length, early_stopping=True)
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

    input_text = collect_text_fields(doc, include_keys=["Narrative","Callback"])
    if not input_text.strip():
        print("Input text is empty after filtering.")
        return

    generated_summary = generate_summary(input_text)
    original_synopsis = get_synopsis(doc)

    rouge = evaluate_rouge(generated_summary, original_synopsis)

    P, R, F1 = bert_score_fn(
        [generated_summary],
        [original_synopsis],
        lang="en",
        verbose=False
    )

    return {
        "acn": acn,
        "input": input_text,
        "generated": generated_summary,
        "target": original_synopsis,
        "rouge": {
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure
        },
        "bert": {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item()
        }
    }
