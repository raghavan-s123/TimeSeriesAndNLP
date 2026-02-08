import pandas as pd
import re

# -----------------------------
# Built-in English stopwords
# -----------------------------
ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by do does doing down during each few for from further had
has have having he her here hers him himself his how i if in into is it its itself me
more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why with you your yours yourself yourselves
""".split())

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    """
    Lowercase, remove URLs, mentions, punctuation, numbers, stopwords
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in ENGLISH_STOPWORDS]
    return " ".join(words)

# -----------------------------
# Multi-label splitter
# -----------------------------
def split_labels(label_string):
    """
    Converts a comma-separated string into a list of labels.
    Returns empty list if input is NaN or empty.
    """
    if pd.isna(label_string) or label_string == "":
        return []
    return [l.strip() for l in label_string.split(",")]

# -----------------------------
# Train/test split
# -----------------------------
def split_data(df, test_ratio=0.2, random_state=42):
    """
    Randomly splits DataFrame into train and test sets
    """
    train = df.sample(frac=1 - test_ratio, random_state=random_state)
    test = df.drop(train.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)

# -----------------------------
# Binary label preparation
# -----------------------------
def create_binary_label(df, column, positive_value):
    """
    Converts a column into binary 0/1 labels.
    positive_value -> 1, all others -> 0
    """
    return df[column].apply(lambda x: 1 if x == positive_value else 0)

# -----------------------------
# Multi-class label encoding
# -----------------------------
def encode_labels(df, column):
    """
    Label encodes a categorical column
    Returns: encoded_series, label_encoder
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(df[column])
    return encoded, le
