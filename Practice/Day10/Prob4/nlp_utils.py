# ==============================
# nlp_utils.py
# ==============================

import re
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove punctuation/special characters
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Split multi-labels
# -----------------------------
def split_labels(label_str):
    """
    Split comma-separated labels into a list.
    Handles NaN values safely.
    """
    if pd.isna(label_str):
        return []
    return [l.strip() for l in label_str.split(",")]

# -----------------------------
# Train/test split
# -----------------------------
def split_data(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets.
    Returns train_df, test_df with reset indexes.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)
