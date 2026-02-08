import pandas as pd
import os
import sys
import warnings
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

warnings.simplefilter("ignore")

# Import NLP utilities
from nlp_utils import clean_text, split_data

# -----------------------------
# Main Function
# -----------------------------
def main():

    # -----------------------------
    # STEP 1: Read dataset
    # -----------------------------
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "Sample.csv"  # default

    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print("Dataset file not found.")
        return

    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    # -----------------------------
    # STEP 2: Train / Test Split
    # -----------------------------
    train_df, test_df = split_data(df)

    # -----------------------------
    # STEP 3: Text Cleaning
    # -----------------------------
    train_df["clean_text"] = train_df["review"].apply(clean_text)
    test_df["clean_text"] = test_df["review"].apply(clean_text)

    # =================================================
    # 3.1 Store fastText Multi-Class Predictions
    # =================================================
    # NOTE: fastText predictions are mocked here
    random.seed(42)
    ft_labels = train_df["sentiment_encoded"].unique().tolist()

    def fasttext_predict_mock(text):
        return random.choice(ft_labels)

    test_df["pred_ft"] = test_df["clean_text"].apply(fasttext_predict_mock)

    # =================================================
    # 3.2 Predict Multi-Class data with sklearn model
    # =================================================
    multi_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", MultinomialNB())
    ])

    multi_pipeline.fit(
        train_df["clean_text"],
        train_df["sentiment_encoded"]
    )

    test_df["pred_sklearn"] = multi_pipeline.predict(
        test_df["clean_text"]
    )

    print("\nMulti-Class sklearn Sample Predictions:")
    print(test_df["pred_sklearn"].head(10).tolist())

    # =================================================
    # 3.3 Create Mock GenAI Predictor
    # =================================================
    genai_labels = train_df["sentiment_encoded"].unique().tolist()

    def genai_predict_mock(text):
        return random.choice(genai_labels)

    test_df["pred_genai"] = test_df["clean_text"].apply(genai_predict_mock)

    # =================================================
    # 3.4 Evaluate all three models
    # =================================================
    y_true = test_df["sentiment_encoded"]

    ft_acc  = accuracy_score(y_true, test_df["pred_ft"])
    sk_acc  = accuracy_score(y_true, test_df["pred_sklearn"])
    gen_acc = accuracy_score(y_true, test_df["pred_genai"])

    print("\n============= MULTI-CLASS ACCURACY =============")
    print("fastText Accuracy :", round(ft_acc, 4))
    print("sklearn Accuracy  :", round(sk_acc, 4))
    print("GenAI Accuracy    :", round(gen_acc, 4))

    # =================================================
    # 3.5 Build Alignment Comparison Table
    # =================================================
    test_df["agree_ft_sk"]  = test_df["pred_ft"] == test_df["pred_sklearn"]
    test_df["agree_ft_gen"] = test_df["pred_ft"] == test_df["pred_genai"]
    test_df["agree_sk_gen"] = test_df["pred_sklearn"] == test_df["pred_genai"]

    alignment_results = test_df[
        ["agree_ft_sk", "agree_ft_gen", "agree_sk_gen"]
    ].mean()

    print("\n============= ALIGNMENT RESULTS =============")
    print(alignment_results)

    # =================================================
    # 3.6 Inspect High-Confidence fastText Predictions
    # =================================================
    # Mock confidence scores
    test_df["ft_confidence"] = [
        round(random.uniform(0.60, 0.99), 3)
        for _ in range(len(test_df))
    ]

    print("\n High-Confidence fastText Predictions:")
    print(
        test_df[test_df["ft_confidence"] > 0.85][
            ["review", "pred_ft", "ft_confidence"]
        ].head(5)
    )

    # =================================================
    # 3.7 Interpretation
    # =================================================
    print("\n============= INTERPRETATION =============")

    print("\n Where fastText > sklearn?")
    print(
        test_df[
            (test_df["pred_ft"] == y_true) &
            (test_df["pred_sklearn"] != y_true)
        ][["review", "pred_ft", "pred_sklearn"]].head(3)
    )

    print("\n Where sklearn > fastText?")
    print(
        test_df[
            (test_df["pred_sklearn"] == y_true) &
            (test_df["pred_ft"] != y_true)
        ][["review", "pred_sklearn", "pred_ft"]].head(3)
    )

    print("\n Where GenAI > both?")
    print(
        test_df[
            (test_df["pred_genai"] == y_true) &
            (test_df["pred_sklearn"] != y_true) &
            (test_df["pred_ft"] != y_true)
        ][["review", "pred_genai", "pred_ft", "pred_sklearn"]].head(3)
    )


if __name__ == "__main__":
    main()
