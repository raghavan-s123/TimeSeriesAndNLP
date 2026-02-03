import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def main():
    # ============================
    # Step 0: Input text filename
    # ============================
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load text file
    # ============================
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    # ============================
    # Step 2: Load spaCy model
    # ============================
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Process text with spaCy
    # ============================
    doc = nlp(content)

    # ============================
    # Step 4: Dependency Parsing
    # Extract sentences where the
    # main subject is a PERSON
    # ============================
    print("=== Sentences with PERSON as Main Subject ===\n")

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "nsubj" and token.ent_type_ == "PERSON":
                print(f"Sentence : {sent.text.strip()}")
                print(f"Subject  : {token.text}")
                print(f"Main Verb: {token.head.text}")
                print("-" * 50)

if __name__ == "__main__":
    main()
