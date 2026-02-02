import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import spacy
from collections import Counter

def main():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return

    text = """
    Sonia O'Sullivan has indicated that she would like to participate at the event.
    The runner has trained at her base and has shown improvement at the nationals.
    """

    doc = nlp(text)

   
    freq_counts = Counter()
    for token in doc:
        freq_counts[token.text] += 1

    targets = ['the', 'at', 'has', '.']
    for t in targets:
        print(f"Frequency of '{t}':", freq_counts[t])

   
    highest = max(targets, key=lambda x: freq_counts[x])
    print("Highest frequency among targets:", highest, "→", freq_counts[highest])


if __name__ == "__main__":
    main()
