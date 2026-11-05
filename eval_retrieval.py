"""Measure retrieval quality and help choose the 'no answer' threshold."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search import NoticeIndex

df = pd.read_csv("data/notices.csv").dropna(subset=["text"])
texts = df["text"].tolist()
qa = pd.read_csv("data/qa_eval.csv")
in_scope = qa[qa["must_contain"].notna()]
out_scope = qa[qa["must_contain"].isna()]

index = NoticeIndex(texts)

# Baseline: TF-IDF keyword retrieval
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
doc_matrix = tfidf.fit_transform(texts)


def tfidf_rank(question, k=3):
    scores = (doc_matrix @ tfidf.transform([question]).T).toarray().ravel()
    return [texts[i] for i in np.argsort(-scores)[:k]]


def contains(text, phrase):
    return phrase.lower() in text.lower()


results = {"Embeddings": [0, 0], "TF-IDF": [0, 0]}
top1_scores = []
for _, row in in_scope.iterrows():
    emb_hits = index.search(row["question"], 3)
    emb_texts = [h["text"] for h in emb_hits]
    tf_texts = tfidf_rank(row["question"])
    for name, ranked in [("Embeddings", emb_texts), ("TF-IDF", tf_texts)]:
        results[name][0] += contains(ranked[0], row["must_contain"])
        results[name][1] += any(contains(t, row["must_contain"]) for t in ranked)
    top1_scores.append(emb_hits[0]["score"])
    if not contains(emb_texts[0], row["must_contain"]):
        print(f"MISS: {row['question']}\n   got: {emb_texts[0][:90]}")

n = len(in_scope)
table = pd.DataFrame(
    [{"method": m, "top1_accuracy": round(v[0] / n, 3), "top3_accuracy": round(v[1] / n, 3)}
     for m, v in results.items()])
print("\n", table.to_string(index=False), f"\n({n} in-scope questions)")

out_scores = [index.search(q, 1)[0]["score"] for q in out_scope["question"]]
print(f"\nIn-scope top-1 similarity:     min={min(top1_scores):.2f}  mean={np.mean(top1_scores):.2f}")
print(f"Out-of-scope top-1 similarity: max={max(out_scores):.2f}  mean={np.mean(out_scores):.2f}")
suggest = (min(top1_scores) + max(out_scores)) / 2
note = "" if min(top1_scores) > max(out_scores) else "  (ranges overlap: favour fewer wrong answers)"
print(f"Suggested MIN_SCORE: about {suggest:.2f}{note}")

Path("results").mkdir(exist_ok=True)
table.to_csv("results/retrieval_comparison.csv", index=False)
