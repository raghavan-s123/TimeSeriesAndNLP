# ==============================
# main.py
# ==============================

import pandas as pd
import os
import sys
import warnings

warnings.simplefilter("ignore")

# Import helper functions from nlp_utils
from nlp_utils import clean_text, split_labels, split_data

# -----------------------------
# Main processing
# -----------------------------
def main():
    # ==============================
    # Step 1: Hardcoded dataset filenames
    # ==============================
    train_file = "Sample.csv"  # You can replace with Excel if needed
    test_file  = "Sample.csv"

    train_path = os.path.join(sys.path[0], train_file)
    test_path  = os.path.join(sys.path[0], test_file)

    # ==============================
    # Step 2: Load datasets
    # ==============================
    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    train_df = train_df.dropna()
    test_df  = test_df.dropna()

    print("\n=== Training Data Sample ===")
    print(train_df.head())
    print("\n=== Test Data Sample ===")
    print(test_df.head())

    # ==============================
    # Step 3: Clean text
    # ==============================
    if "clean_text" not in train_df.columns:
        if "review" in train_df.columns:
            train_df["clean_text"] = train_df["review"].apply(clean_text)
            test_df["clean_text"]  = test_df["review"].apply(clean_text)
        else:
            print("No column 'clean_text' or 'review' found.")
            sys.exit(1)

    # ==============================
    # Step 4: Prepare fastText-ready Binary Data
    # ==============================
    if "binary_sentiment" in train_df.columns:
        train_df['ft_label_binary'] = '__label__' + train_df['binary_sentiment'].astype(str)
        train_df['ft_format_binary'] = train_df['ft_label_binary'] + " " + train_df['clean_text']
        print("\n=== Binary FastText Sample ===")
        print("\n".join(train_df['ft_format_binary'].head().tolist()))

        out_path = os.path.join(sys.path[0], "train_fasttext_bn.txt")
        train_df['ft_format_binary'].to_csv(out_path, index=False, header=False)

    # ==============================
    # Step 5: Prepare fastText-ready Multi-Class Data
    # ==============================
    if "sentiment" in train_df.columns:
        train_df['ft_label_multiclass'] = '__label__' + train_df['sentiment'].astype(str)
        test_df['ft_label_multiclass']  = '__label__' + test_df['sentiment'].astype(str)

        train_df['ft_format_multiclass'] = train_df['ft_label_multiclass'] + " " + train_df['clean_text']
        test_df['ft_format_multiclass']  = test_df['ft_label_multiclass'] + " " + test_df['clean_text']

        print("\n=== Multi-Class FastText Sample ===")
        print("\n".join(train_df['ft_format_multiclass'].head().tolist()))

        out_path_mc = os.path.join(sys.path[0], "train_fasttext_mc.txt")
        train_df['ft_format_multiclass'].to_csv(out_path_mc, index=False, header=False)

    # ==============================
    # Step 6: Prepare fastText-ready Multi-Label Data
    # ==============================
    if "emotion_labels" in train_df.columns:
        def convert_labels(row):
            labels = split_labels(row['emotion_labels'])
            labels = ['__label__' + l for l in labels]
            return " ".join(labels)

        train_df['ft_label_multi'] = train_df.apply(convert_labels, axis=1)
        train_df['ft_format_multi'] = train_df['ft_label_multi'] + " " + train_df['clean_text']

        print("\n=== Multi-Label FastText Sample ===")
        print("\n".join(train_df['ft_format_multi'].head().tolist()))

        out_path_ml = os.path.join(sys.path[0], "train_fasttext_ml.txt")
        train_df['ft_format_multi'].to_csv(out_path_ml, index=False, header=False)


if __name__ == "__main__":
    main()
