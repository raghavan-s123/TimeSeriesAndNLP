# ==============================
# main.py
# ==============================

import pandas as pd
import os
import sys
import warnings
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

warnings.simplefilter("ignore")

# Import functions from nlp_utils
from nlp_utils import clean_text, split_labels, split_data

# -----------------------------
# Main processing
# -----------------------------
def main():
    # Step 1: Get dataset filename
    filename = input("Enter dataset filename (CSV or Excel): ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # Step 2: Load dataset
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print("\n=== First 5 Rows ===")
    print(df.head())
    print(f"\nNumber of samples: {df.shape[0]}")
    print("\n=== Data Types ===")
    print(df.dtypes)

    # Step 3: Train/test split
    train, test = split_data(df)
    print(f"\nTrain: {train.shape[0]}, Test: {test.shape[0]}")

    # Step 4: Clean review text
    if "review" not in train.columns:
        print("Column 'review' not found — cannot clean text.")
        sys.exit(1)

    train["cleaned_text"] = train["review"].apply(clean_text)
    test["cleaned_text"] = test["review"].apply(clean_text)

    print("\n=== Sample Cleaned Text ===")
    print(train[["review", "cleaned_text"]].head())

    # Step 5: TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(train["cleaned_text"])
    X_test_tfidf = tfidf.transform(test["cleaned_text"])
    print("\n=== TF-IDF Shapes ===")
    print("Train:", X_train_tfidf.shape, "Test:", X_test_tfidf.shape)

    # Save TF-IDF vectorizer
    tfidf_file = os.path.join(sys.path[0], "tfidf.pkl")
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidf, f)
    

    # Step 6: Label encoding (sentiment)
    if "sentiment" in train.columns:
        le_sentiment = LabelEncoder()
        train["sentiment_encoded"] = le_sentiment.fit_transform(train["sentiment"])
        test["sentiment_encoded"] = le_sentiment.transform(test["sentiment"])
        print("\n=== Sentiment Mapping ===")
        print(dict(zip(le_sentiment.classes_, le_sentiment.transform(le_sentiment.classes_))))

    # Step 7: Multi-label encoding (emotion_labels)
    if "emotion_labels" in train.columns:
        train["emotion_list"] = train["emotion_labels"].apply(split_labels)
        test["emotion_list"] = test["emotion_labels"].apply(split_labels)

        mlb = MultiLabelBinarizer()
        Y_train_mlabel = mlb.fit_transform(train["emotion_list"])
        Y_test_mlabel = mlb.transform(test["emotion_list"])
        print("\n=== Multi-label Classes ===")
        print(mlb.classes_)
        print("Multi-label shape (train):", Y_train_mlabel.shape)


if __name__ == "__main__":
    main()
