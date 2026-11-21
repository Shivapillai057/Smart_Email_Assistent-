"""
preload_models.py
-----------------
Run this ONCE before starting the app:

    python preload_models.py

What it does:
  1. Trains TF-IDF + Logistic Regression on email_dataset_400k.csv
     and saves it to  → spam_model_cache.pkl
  2. Downloads facebook/bart-large-cnn from HuggingFace
     and saves it to  → bart_model_cache/

After this, app.py loads both instantly from disk — no internet needed.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration

# ── Paths ────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PARENT       = os.path.dirname(HERE)
DATASET_PATH = os.path.join(PARENT, "email_dataset_400k.csv")
SPAM_CACHE   = os.path.join(HERE,   "spam_model_cache.pkl")
BART_CACHE   = os.path.join(HERE,   "bart_model_cache")

# ── 1. SPAM MODEL ─────────────────────────────────────────────────────────────
def preload_spam_model():
    if os.path.exists(SPAM_CACHE):
        print(f"[SKIP] spam_model_cache.pkl already exists — skipping training.")
        return

    print("[1/4] Reading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df[["raw_text", "label"]]

    print("[2/4] Preprocessing...")
    df = df.drop_duplicates()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    X = df["raw_text"]
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[3/4] Training TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_features=50000)
    X_train_tfidf = tfidf.fit_transform(X_train)

    log_reg = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    log_reg.fit(X_train_tfidf, y_train)

    print("[4/4] Saving to spam_model_cache.pkl ...")
    with open(SPAM_CACHE, "wb") as f:
        pickle.dump((tfidf, log_reg), f)

    print("   ✅ Spam model saved!\n")


# ── 2. BART MODEL ─────────────────────────────────────────────────────────────
def preload_bart_model():
    if os.path.exists(BART_CACHE) and os.listdir(BART_CACHE):
        print(f"[SKIP] bart_model_cache/ already exists — skipping download.")
        return

    os.makedirs(BART_CACHE, exist_ok=True)

    print("[1/2] Downloading BART tokenizer (facebook/bart-large-cnn)...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.save_pretrained(BART_CACHE)
    print("      Tokenizer saved.")

    print("[2/2] Downloading BART model weights (~1.6 GB, please wait)...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model.save_pretrained(BART_CACHE)
    print("   ✅ BART model saved to bart_model_cache/\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Smart Email Assistant — Model Preloader")
    print("=" * 55)

    print("\n── STEP 1: Spam Classifier ─────────────────────────")
    preload_spam_model()

    print("\n── STEP 2: BART Summarization Model ────────────────")
    preload_bart_model()

    print("=" * 55)
    print("   All models ready! Now run:  streamlit run app.py")
    print("=" * 55)
