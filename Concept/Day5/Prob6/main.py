import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow INFO/WARNING messages

import sys
import warnings
warnings.simplefilter(action='ignore')  # suppress other warnings

import spacy

def main():
  
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\nOriginal Text Sample:")
    print(content[:300])
    print()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    
    sentences = [
        "All is well that ends well.",
        "Apple is looking at buying a U.K. startup for $ 1 billion dollars.",
        "Time flies like an arrow.",
        "The monkey ate the banana before I could stop him.",
        content 
    ]

    for i, sent in enumerate(sentences, 1):
        print(f"\n=== POS Tagging for Sentence {i} ===")
        doc = nlp(sent)
        print("Word\t\tPOS\t\tTag")
        print("----------------------------------")
        for token in doc:
            if not token.is_space:
                print(f"{token.text}\t\t{token.pos_}\t\t{token.tag_}")
        print()

if __name__ == "__main__":
    main()
