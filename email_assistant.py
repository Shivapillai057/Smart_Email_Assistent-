"""
email_assistant.py
------------------
Backend logic for the Smart Email Assistant.
Extracted and refactored from RR2.ipynb — no functionality changed.

Responsibilities:
  1. Train and cache the TF-IDF + Logistic Regression spam classifier.
  2. Load the BART summarization model (facebook/bart-large-cnn).
  3. Provide `process_email()` as the single public entry point for the UI.
"""

import re
import os
import pickle

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PARENT       = os.path.dirname(_HERE)                    # …/smart email/
_DATASET_PATH = os.path.join(_PARENT, "email_dataset_400k.csv")
_CACHE_PATH   = os.path.join(_HERE,   "spam_model_cache.pkl")
_BART_CACHE   = os.path.join(_HERE,   "bart_model_cache")  # saved by preload_models.py

SPAM_THRESHOLD = 0.80          # strict threshold — do NOT change
BART_MODEL_ID  = "facebook/bart-large-cnn"

# ---------------------------------------------------------------------------
# 1. SPAM DETECTION — TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def _train_spam_model():
    """
    Replicates the exact preprocessing + training pipeline from RR2.ipynb.
    Returns (tfidf, log_reg) fitted objects.
    """
    df = pd.read_csv(_DATASET_PATH)

    # Keep only the two columns we need (cell 5 in notebook)
    df = df[["raw_text", "label"]]

    # Drop duplicates (cell 8)
    df = df.drop_duplicates()

    # Fix labels — keep only 0 (HAM) and 1 (SPAM) (cell 11)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    X = df["raw_text"]
    y = df["label"]

    # Exact TF-IDF settings from cell 14
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=50000,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Exact LogisticRegression settings from cell 15
    log_reg = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    log_reg.fit(X_train_tfidf, y_train)

    return tfidf, log_reg


def load_spam_model():
    """
    Returns (tfidf, log_reg). Loads from cache if available; trains otherwise.
    """
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "rb") as f:
            tfidf, log_reg = pickle.load(f)
        return tfidf, log_reg

    tfidf, log_reg = _train_spam_model()
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump((tfidf, log_reg), f)
    return tfidf, log_reg


# ---------------------------------------------------------------------------
# 2. BART SUMMARIZATION — facebook/bart-large-cnn
# ---------------------------------------------------------------------------

def load_bart_model():
    """
    Loads and returns (tokenizer, model, device).
    Loads from local bart_model_cache/ folder if preload_models.py was run,
    otherwise falls back to downloading from HuggingFace.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use local cache if available (fastest — no internet needed)
    source = _BART_CACHE if (os.path.exists(_BART_CACHE) and os.listdir(_BART_CACHE)) else BART_MODEL_ID

    tokenizer = BartTokenizer.from_pretrained(source)
    model     = BartForConditionalGeneration.from_pretrained(source)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_summary(email_text: str, tokenizer, model, device) -> str:
    """
    Generates an abstractive summary with BART.
    Exact same logic as `generate_summary()` in cell 38 of RR2.ipynb.
    Short emails (< 20 words) bypass BART to save compute.
    """
    word_count = len(email_text.split())
    if word_count < 20:
        return email_text.strip()   # already concise

    inputs = tokenizer(
        email_text,
        max_length=1024,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=100,
            min_length=20,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ---------------------------------------------------------------------------
# 3. DATE / TIME EXTRACTION — re (cell 38 in RR2.ipynb)
# ---------------------------------------------------------------------------

def extract_reminder_info(text: str) -> list:
    """
    Extracts reminder sentences that contain a date/day/time AND a keyword.
    Exact same regex patterns as `extract_reminder_info()` in RR2.ipynb.
    Returns a list of dicts: {date, day, time, details}.
    """
    patterns = {
        "date": (
            r"\b(?:"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r"|\d{4}-\d{1,2}-\d{1,2}"
            r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
            r"(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
            r")\b"
        ),
        "day": (
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
            r"|today|tomorrow|day after tomorrow)\b"
        ),
        "time": (
            r"\b(?:(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:AM|PM)?"
            r"|(?:0?[1-9]|1[0-2])\s*(?:AM|PM)"
            r"|noon|midnight)\b"
        ),
    }

    keywords = (
        r"\b(deadline|due|submit|submission|meeting|interview|appointment"
        r"|reminder|register|registration|schedule|expires|payment|exam"
        r"|event|call|before|by)\b"
    )

    reminders = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        dates = re.findall(patterns["date"], sentence, re.I)
        days  = re.findall(patterns["day"],  sentence, re.I)
        times = re.findall(patterns["time"], sentence, re.I)

        if (dates or days or times) and re.search(keywords, sentence, re.I):
            reminders.append({
                "date":    list(dict.fromkeys(dates)),
                "day":     list(dict.fromkeys(days)),
                "time":    list(dict.fromkeys(times)),
                "details": sentence,
            })

    return reminders


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE — single public function
# ---------------------------------------------------------------------------

def process_email(
    raw_text: str,
    tfidf,
    log_reg,
    tokenizer,
    bart_model,
    device,
    threshold: float = SPAM_THRESHOLD,   # adjustable from UI slider
) -> dict:
    """
    Full email processing pipeline — replicates `process_incoming_email()`
    from cell 38 of RR2.ipynb.

    Returns a dict:
    {
        "status":           "SPAM" | "HAM",
        "spam_probability": float (0–100),
        "threshold_used":   float (0–100),
        "is_short":         bool,
        "summary":          str | None,
        "reminders":        list[dict] | [],
    }
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Email text cannot be empty.")

    # --- Phase 1: Spam Detection ---
    email_vector     = tfidf.transform([raw_text])
    spam_probability = log_reg.predict_proba(email_vector)[0][1]

    if spam_probability >= threshold:
        return {
            "status":           "SPAM",
            "spam_probability": round(spam_probability * 100, 2),
            "threshold_used":   round(threshold * 100, 2),
            "is_short":         False,
            "summary":          None,
            "reminders":        [],
        }

    # --- Phase 2: Summarization ---
    word_count = len(raw_text.split())
    is_short   = word_count < 20
    summary    = generate_summary(raw_text, tokenizer, bart_model, device)

    # --- Phase 3: Date / Time Extraction ---
    reminders = extract_reminder_info(raw_text)

    return {
        "status":           "HAM",
        "spam_probability": round(spam_probability * 100, 2),
        "threshold_used":   round(threshold * 100, 2),
        "is_short":         is_short,
        "summary":          summary,
        "reminders":        reminders,
    }
