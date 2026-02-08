import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.simplefilter(action='ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from nlp_utils import clean_text, split_labels


def main():
    filename = input("").strip()
    file_path = os.path.join(sys.path[0], filename)

    # Load dataset
    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format")
        return

    print("=== First 5 Rows ===")
    print(df.head())

    print(f"\nNumber of samples: {df.shape[0]}")

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    # Clean text
    if 'text' not in df.columns:
        print("Column 'text' not found")
        return

    df['clean_text'] = df['text'].apply(clean_text)
    print("\n=== Sample Cleaned Text ===")
    print(df[['text', 'clean_text']].head())

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    print("\nTF-IDF Shape:", X_tfidf.shape)

    # Sentiment Encoding
    if 'sentiment' in df.columns:
        le = LabelEncoder()
        df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
        print("\nSentiment Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Emotion Multi-label Encoding
    if 'emotion_labels' in df.columns:
        mlb = MultiLabelBinarizer()
        df['emotion_list'] = df['emotion_labels'].apply(split_labels)
        emotion_encoded = mlb.fit_transform(df['emotion_list'])

        print("\nEmotion Classes:", mlb.classes_)
        print("Emotion Encoding Shape:", emotion_encoded.shape)

if __name__ == "__main__":
    main()
