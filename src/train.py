import os, json, joblib
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from src.preprocessing import preprocess
from src.vectorizer import get_vectorizer
from src.evaluate import evaluate

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METHODS = ["tfidf", "word2vec", "glove"]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading IMDB dataset...")
    ds = load_dataset("imdb")
    train_texts = [preprocess(t) for t in ds["train"]["text"]]
    train_labels = np.array(ds["train"]["label"])
    test_texts = [preprocess(t) for t in ds["test"]["text"]]
    test_labels = np.array(ds["test"]["label"])

    results = {}
    for name in METHODS:
        print(f"\n=== {name.upper()} ===")
        vec = get_vectorizer(name)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, train_labels)

        res = evaluate(clf, X_test, test_labels)
        results[name] = {k: v for k, v in res.items() if k != "predictions"}
        print(f"Accuracy: {res['accuracy']:.4f}  Correct: {res['correct']}  Incorrect: {res['incorrect']}")

        joblib.dump({"model": clf, "vectorizer": vec}, os.path.join(MODELS_DIR, f"{name}.pkl"))

    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll models saved to models/")


if __name__ == "__main__":
    main()
