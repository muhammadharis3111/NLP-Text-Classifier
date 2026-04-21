import os, joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import preprocess
from src.evaluate import evaluate

MODELS_DIR = "models"
METHODS = ["tfidf", "word2vec", "glove"]
LABEL_MAP = {0: "Negative", 1: "Positive"}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


def predict(name, texts_clean):
    bundle = load_model(name)
    X = bundle["vectorizer"].transform(texts_clean)
    return bundle["model"].predict(X)


st.set_page_config(page_title="NLP Text Classifier", layout="wide")
st.title("NLP Text Classification & Cross-Dataset Testing")

mode = st.sidebar.radio("Mode", ["Single Model", "Compare All Models"])
uploaded = st.file_uploader("Upload test CSV (columns: `text`, optionally `label`)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must have a `text` column.")
        st.stop()

    has_labels = "label" in df.columns
    st.info(f"Loaded {len(df)} samples. Labels {'found' if has_labels else 'not found'}.")

    texts_clean = [preprocess(t) for t in df["text"]]
    labels = np.array(df["label"]) if has_labels else None

    if mode == "Single Model":
        choice = st.sidebar.selectbox("Vectorizer", METHODS)
        preds = predict(choice, texts_clean)
        df["prediction"] = [LABEL_MAP[p] for p in preds]

        if has_labels:
            res = evaluate(load_model(choice)["model"],
                           load_model(choice)["vectorizer"].transform(texts_clean), labels)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{res['accuracy']:.2%}")
            c2.metric("Correct", res["correct"])
            c3.metric("Incorrect", res["incorrect"])

            fig, ax = plt.subplots()
            ax.bar(["Correct", "Incorrect"], [res["correct"], res["incorrect"]], color=["#2ecc71", "#e74c3c"])
            st.pyplot(fig)

            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        st.dataframe(df)
        st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")

    else:
        rows = []
        all_preds = {}
        for name in METHODS:
            preds = predict(name, texts_clean)
            all_preds[name] = preds
            if has_labels:
                res = evaluate(load_model(name)["model"],
                               load_model(name)["vectorizer"].transform(texts_clean), labels)
                rows.append({"Method": name.upper(), "Accuracy": f"{res['accuracy']:.2%}",
                              "Correct": res["correct"], "Incorrect": res["incorrect"]})
            else:
                pos = int(np.sum(preds == 1))
                rows.append({"Method": name.upper(), "Positive": pos, "Negative": len(preds) - pos})

        st.subheader("Comparison Table")
        st.table(pd.DataFrame(rows))

        if has_labels:
            fig, ax = plt.subplots()
            x = np.arange(len(METHODS))
            accs = [float(r["Accuracy"].strip("%")) / 100 for r in rows]
            ax.bar(x, accs, tick_label=[m.upper() for m in METHODS], color=["#3498db", "#9b59b6", "#e67e22"])
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.subheader("Confusion Matrices")
            cm_cols = st.columns(len(METHODS))
            for i, name in enumerate(METHODS):
                with cm_cols[i]:
                    res = evaluate(load_model(name)["model"],
                                   load_model(name)["vectorizer"].transform(texts_clean), labels)
                    st.write(f"**{name.upper()}**")
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Negative", "Positive"],
                                yticklabels=["Negative", "Positive"], ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

        out = df.copy()
        for name in METHODS:
            out[f"pred_{name}"] = [LABEL_MAP[p] for p in all_preds[name]]
        st.dataframe(out)
        st.download_button("Download Results", out.to_csv(index=False), "comparison_results.csv", "text/csv")
else:
    st.info("Upload a CSV file to get started.")
