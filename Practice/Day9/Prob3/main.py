import pandas as pd
import os
import sys
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.simplefilter("ignore")

# Import NLP utilities
from nlp_utils import clean_text, split_labels, split_data

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
        filename = "Sample.csv"  # default for platform

    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print("No dataset file found.")
        return

    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    # -----------------------------
    # STEP 2: Train/Test Split
    # -----------------------------
    train, test = split_data(df)

    # -----------------------------
    # STEP 3: Text Cleaning
    # -----------------------------
    if "review" not in df.columns:
        print("Column 'review' not found.")
        return

    train["cleaned_text"] = train["review"].apply(clean_text)
    test["cleaned_text"] = test["review"].apply(clean_text)

    # -----------------------------
    # STEP 4: TF-IDF
    # -----------------------------
    tfidf = TfidfVectorizer(max_features=3000)
    X_train = tfidf.fit_transform(train["cleaned_text"])
    X_test = tfidf.transform(test["cleaned_text"])

    # =================================================
    # 3.1 BINARY CLASSIFICATION
    # =================================================
    if "binary_sentiment" in train.columns:
        # Convert to numeric 0/1
        train["binary_sentiment_num"] = train["binary_sentiment"].map({"negative":0, "positive":1})
        test["binary_sentiment_num"]  = test["binary_sentiment"].map({"negative":0, "positive":1})

        binary_model = LogisticRegression(max_iter=1000)
        binary_model.fit(X_train, train["binary_sentiment_num"])
        binary_preds = binary_model.predict(X_test)

        print("\n============= BINARY MODEL =============")
        print("Accuracy :", accuracy_score(test["binary_sentiment_num"], binary_preds))
        print("Precision:", precision_score(test["binary_sentiment_num"], binary_preds, zero_division=0))
        print("Recall   :", recall_score(test["binary_sentiment_num"], binary_preds, zero_division=0))
        print("F1 Score :", f1_score(test["binary_sentiment_num"], binary_preds, zero_division=0))

    # =================================================
    # 3.2 MULTI-CLASS CLASSIFICATION
    # =================================================
    if "sentiment_encoded" in train.columns:
        multi_model = MultinomialNB()
        multi_model.fit(X_train, train["sentiment_encoded"])
        multi_preds = multi_model.predict(X_test)

        print("\n============= MULTI-CLASS MODEL =============")
        print("Accuracy :", accuracy_score(test["sentiment_encoded"], multi_preds))
        print("Confusion Matrix:")
        print(confusion_matrix(test["sentiment_encoded"], multi_preds))
        print("Classification Report:")
        print(classification_report(test["sentiment_encoded"], multi_preds, zero_division=0))

    # =================================================
    # 3.3 MULTI-LABEL CLASSIFICATION
    # =================================================
    if "emotion_labels" in train.columns:
        train["emotion_list"] = train["emotion_labels"].apply(split_labels)
        test["emotion_list"] = test["emotion_labels"].apply(split_labels)

        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(train["emotion_list"])
        Y_test = mlb.transform(test["emotion_list"])

        multilabel_model = OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )
        multilabel_model.fit(X_train, Y_train)
        multilabel_preds = multilabel_model.predict(X_test)

        print("\n============= MULTI-LABEL MODEL =============")
        print("Micro F1 :", f1_score(Y_test, multilabel_preds, average="micro", zero_division=0))
        print("Macro F1 :", f1_score(Y_test, multilabel_preds, average="macro", zero_division=0))

        print("\nPer-Label F1 Scores:")
        per_label_f1 = f1_score(Y_test, multilabel_preds, average=None, zero_division=0)
        for label, score in zip(mlb.classes_, per_label_f1):
            print(f"{label}: {score:.4f}")

    # =================================================
    # 3.4 Metric Interpretation
    # =================================================
    print("""
Metric Interpretation:
Accuracy  : Overall correctness
Precision : Correct positive predictions
Recall    : Actual positives captured
F1 Score  : Balance of precision and recall
Micro F1  : Emphasizes frequent labels
Macro F1  : Treats all labels equally
""")


if __name__ == "__main__":
    main()
