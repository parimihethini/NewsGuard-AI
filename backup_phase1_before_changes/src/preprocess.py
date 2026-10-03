import re
import string
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import (
    FAKE_PATH,
    TRUE_PATH,
    MAX_WORDS,
    MAX_LEN,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
    TOKENIZER_PATH,
)

# Download NLTK data (safe if already downloaded)
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Canonical text cleaning used across all phases:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation
    4. Remove digits
    5. Filter English stopwords and short tokens (len <= 2)
    6. Lemmatize words with WordNetLemmatizer
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def encode_texts(texts, tokenizer, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Unified text encoding pipeline:
    raw text(s) -> clean_text -> tokenizer.texts_to_sequences -> pad_sequences.
    Accepts a single string or an iterable of strings.
    Guarantees that inference, LIME, and SHAP use the identical preprocessing as training.
    """
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (pd.Series, np.ndarray)):
        texts = texts.tolist()
    else:
        texts = list(texts)

    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pads = pad_sequences(
        seqs,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )
    return pads


def load_tokenizer(path: str = None):
    """Load pickled Tokenizer instance."""
    target_path = path or TOKENIZER_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Tokenizer file not found at: {target_path}")
    with open(target_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def save_tokenizer(tokenizer, path: str = None):
    """Save Tokenizer instance using pickle."""
    target_path = path or TOKENIZER_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with open(target_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {target_path}")


def load_and_prepare_data(split_mode: str = "3way", deduplicate: bool = True):
    """
    Load Fake.csv & True.csv, deduplicate, label, clean, and perform stratified splits.

    split_mode:
        '3way': Returns (X_train_pad, y_train), (X_val_pad, y_val), (X_test_pad, y_test),
                tokenizer, (raw_test_df)
        '2way': Returns X_train_pad, X_test_pad, y_train, y_test, tokenizer (legacy)
    """
    print(f"Loading FAKE news from: {FAKE_PATH}")
    fake_df = pd.read_csv(FAKE_PATH)
    fake_df["label"] = 1  # FAKE

    print(f"Loading TRUE news from: {TRUE_PATH}")
    true_df = pd.read_csv(TRUE_PATH)
    true_df["label"] = 0  # REAL

    print(f"Initial raw shapes -> Fake: {fake_df.shape}, True: {true_df.shape}")

    # Combine into single DataFrame
    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    total_raw = len(df)

    if not {"title", "text"}.issubset(df.columns):
        raise ValueError(f"Expected 'title' and 'text' columns, but got: {df.columns}")

    # Handle duplicates and empty texts if deduplicate is True
    if deduplicate:
        df["full_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
        non_empty_mask = df["full_text"].str.len() > 10
        df = df[non_empty_mask].copy()

        initial_count = len(df)
        df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
        dedup_count = len(df)
        print(f"Deduplication: removed {total_raw - dedup_count} invalid/duplicate records ({total_raw} -> {dedup_count}).")
    else:
        df["full_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("Cleaning text with canonical clean_text pipeline...")
    df["clean_text"] = df["full_text"].apply(clean_text)

    # Filter any row that became completely empty after cleaning
    valid_mask = df["clean_text"].str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    X_clean = np.array(df["clean_text"].tolist(), dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)

    if split_mode == "3way":
        # 70% Train, 15% Validation, 15% Test
        print("Performing stratified 3-way split: 70% Train, 15% Validation, 15% Test...")
        # Step 1: Separate Test set (15%)
        indices = np.arange(len(df))
        X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
            X_clean, y, indices, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
        )

        # Step 2: Separate Validation set (15% of total = 0.15 / 0.85 of trainval)
        val_relative_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_relative_ratio, random_state=RANDOM_STATE, stratify=y_trainval
        )

        print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Tokenizer fitted ONLY on X_train to prevent leakage
        print(f"Fitting tokenizer on training set only (max_words={MAX_WORDS})...")
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        # Transform sequences
        X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
        X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAX_LEN, padding="post", truncating="post")
        X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post", truncating="post")

        test_df_raw = df.iloc[idx_test].copy()

        return (X_train_pad, y_train), (X_val_pad, y_val), (X_test_pad, y_test), tokenizer, test_df_raw

    else:
        # Legacy 2-way split
        print("Performing legacy 2-way split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
        )
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
        X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post", truncating="post")

        return X_train_pad, X_test_pad, y_train, y_test, tokenizer

