import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def main():
    
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: File not found")
        sys.exit(1)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(content)

    spacy_stopwords = STOP_WORDS

    spacy_stopwords |= {"officially", "announced", "present", "run"}

    spacy_stopwords -= {"hence", "every", "he"}

  
    filtered_tokens = [
        token.lemma_.lower()
        for token in doc
        if token.text.lower() not in spacy_stopwords
        and not token.is_punct
        and not token.is_space
    ]

    print("Filtered Tokens (First 20):")
    print(filtered_tokens[:20])
    print()

    
    cleaned_text = " ".join(filtered_tokens)

    print("Cleaned Text Sample:")
    print(cleaned_text[:200])

if __name__ == "__main__":
    main()
