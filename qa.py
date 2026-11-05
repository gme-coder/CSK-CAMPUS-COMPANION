"""Grounded question answering: retrieve, refuse if unsure, then answer with sources."""
import os
from datetime import date

import requests
from dotenv import load_dotenv

from src.extract import extract_all
from src.search import NoticeIndex

load_dotenv()

# Calibrate with `python -m src.eval_retrieval`, then update this value.
MIN_SCORE = 0.30

SYSTEM_PROMPT = (
    "You are a campus assistant for university students. Answer the question using "
    "ONLY the numbered notices provided. Cite the notice number like [1]. "
    "Be short and clear: state the date, time, place and what the student must do "
    "when they are in the notices. If the notices do not contain the answer, say "
    "you could not find it. Never invent details."
)


def _call_llm(question, hits, today):
    url, key, model = (os.getenv("LLM_API_URL"), os.getenv("LLM_API_KEY"),
                       os.getenv("LLM_MODEL"))
    if not (url and key and model):
        return None
    notices = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    user = (f"Today's date: {today.strftime('%A, %d %B %Y')}\n\n"
            f"Notices:\n{notices}\n\nQuestion: {question}")
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0.1,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user}]},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:                        # network, quota, bad key, anything
        print(f"[LLM unavailable, using fallback] {e}")
        return None


def _fallback_answer(hits, today):
    best = hits[0]
    info = extract_all(best["text"], today)
    details = ", ".join(f"{k}: {v}" for k, v in info.items() if v)
    return f"Best matching notice: {best['text']}\nDetails found: {details}"


def answer(question: str, index: NoticeIndex, k: int = 3,
           min_score: float = MIN_SCORE, today=None):
    today = today or date.today()
    hits = index.search(question, k)
    hits = [h for h in hits if h["score"] >= min_score]
    if not hits:
        return {"answer": "I couldn't find anything about that in the notices I have.",
                "sources": [], "mode": "no_match"}
    text = _call_llm(question, hits, today)
    mode = "llm" if text else "retrieval"
    return {"answer": text or _fallback_answer(hits, today),
            "sources": hits, "mode": mode}


if __name__ == "__main__":
    idx = NoticeIndex()
    print("Ask about campus notices (Ctrl+C to quit)")
    while True:
        q = input("\n> ")
        res = answer(q, idx)
        print(f"\n[{res['mode']}] {res['answer']}")
        for s in res["sources"]:
            print(f"   source ({s['score']:.2f}): {s['text'][:90]}...")
