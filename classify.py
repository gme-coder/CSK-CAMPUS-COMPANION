"""Train and compare two notice classifiers, then save the best one."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline

DATA = Path("data/notices.csv")
RESULTS = Path("results")
MODELS = Path("models")
ENCODER_NAME = "all-MiniLM-L6-v2"

_encoder = None


def get_encoder():
    """Load the sentence-transformer once and reuse it."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder


def train_and_compare():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA).dropna(subset=["text", "category"]).reset_index(drop=True)
    X, y = df["text"].tolist(), df["category"].tolist()
    labels = sorted(set(y))
    print(f"{len(df)} notices, {len(labels)} classes")
    print(df["category"].value_counts(), "\n")

    # Adaptive folds: never more folds than the smallest class has rows
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    print(f"Using {n_splits}-fold cross-validation (smallest class has {min_class} rows)")

    # One clean partition for predictions and the confusion matrix
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Repeated runs with different shuffles give a more stable score on small data
    cv_repeat = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=42)

    # Model A: TF-IDF + Logistic Regression
    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )

    # Model B: sentence embeddings + Logistic Regression
    print("Encoding notices (first run downloads the model, about 90 MB)...")
    emb = get_encoder().encode(X, normalize_embeddings=True, show_progress_bar=True)
    emb_model = LogisticRegression(max_iter=1000)

    candidates = {
        "TF-IDF + LogReg": (tfidf_model, X),
        "Embeddings + LogReg": (emb_model, emb),
    }

    summary, preds = [], {}
    for name, (model, features) in candidates.items():
        pred = cross_val_predict(model, features, y, cv=cv)
        f1_folds = cross_val_score(model, features, y, cv=cv_repeat, scoring="f1_macro")
        preds[name] = pred
        summary.append({
            "model": name,
            "accuracy": round(accuracy_score(y, pred), 3),
            "macro_f1": round(float(f1_folds.mean()), 3),
            "f1_std": round(float(f1_folds.std()), 3),
        })
        print(f"\n===== {name} =====")
        print(classification_report(y, pred, zero_division=0))

        ConfusionMatrixDisplay.from_predictions(
            y, pred, labels=labels, xticks_rotation=45, cmap="Blues")
        plt.title(name)
        plt.tight_layout()
        fname = name.lower().replace(" + ", "_").replace(" ", "_")
        plt.savefig(RESULTS / f"confusion_{fname}.png", dpi=150)
        plt.close()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\n", summary_df.to_string(index=False))

    # Pick the winner by repeated-CV macro F1, then retrain it on ALL the data
    best = summary_df.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    print(f"\nBest model: {best}")
    model, features = candidates[best]
    model.fit(features, y)
    kind = "tfidf" if best.startswith("TF-IDF") else "embedding"
    joblib.dump({"kind": kind, "model": model, "labels": labels},
                MODELS / "classifier.joblib")

    # Save the mistakes for error analysis in the README
    wrong = df.assign(predicted=preds[best])
    wrong = wrong[wrong["category"] != wrong["predicted"]]
    wrong[["text", "category", "predicted"]].to_csv(
        RESULTS / "misclassified.csv", index=False)
    print(f"{len(wrong)} misclassified notices saved to results/misclassified.csv")


def predict_category(text: str) -> str:
    """Used by the app: returns the predicted category for one notice."""
    bundle = joblib.load(MODELS / "classifier.joblib")
    if bundle["kind"] == "tfidf":
        return bundle["model"].predict([text])[0]
    vec = get_encoder().encode([text], normalize_embeddings=True)
    return bundle["model"].predict(vec)[0]


if __name__ == "__main__":
    train_and_compare()
