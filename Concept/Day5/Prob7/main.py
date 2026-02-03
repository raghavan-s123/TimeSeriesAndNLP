import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
import pandas as pd
from spacy import displacy

def main():
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Install using: python -m spacy download en_core_web_sm")
        sys.exit(1)

    text_1 = "The dollar has hit its highest level against the euro after the Federal Reserve head said the US trade deficit is set to stabilize."
    doc_1 = nlp(text_1)

    print("=== 7.2.1 NLP Processed Tokens ===")
    for token in doc_1:
        print(token.text, token.pos_, token.dep_)
    print()

    ner_list = [(ent.text, ent.label_) for ent in doc_1.ents]

    print("=== 7.2.2 Named Entities (Tuples) ===")
    print(ner_list)
    print()

    ner_df = pd.DataFrame(ner_list, columns=["Entity", "Label"])
    print("=== 7.2.3 Named Entities DataFrame ===")
    print(ner_df)
    print()

    doc_full = nlp(content)
    full_ner = [(ent.text, ent.label_) for ent in doc_full.ents]
    full_ner_df = pd.DataFrame(full_ner, columns=["Entity", "Label"])

    print("=== 7.2.4 First 5 Named Entities from File ===")
    print(full_ner_df.head())
    print()

    text_2 = "Taylor Swift will perform in Tokyo next Friday, just after launching her new album with Universal Music Group."
    doc_2 = nlp(text_2)

    ner_q = [(ent.text, ent.label_) for ent in doc_2.ents]
    ner_q_df = pd.DataFrame(ner_q, columns=["Entity", "Label"])

    print("=== Question-based NER DataFrame ===")
    print(ner_q_df)
    print()

    person = [ent.text for ent in doc_2.ents if ent.label_ == "PERSON"]
    date = [ent.text for ent in doc_2.ents if ent.label_ == "DATE"]
    place = [ent.text for ent in doc_2.ents if ent.label_ == "GPE"]
    company = [ent.text for ent in doc_2.ents if ent.label_ == "ORG"]

    print("=== Extracted Answers ===")
    print("Who is the person performing?:", person[0] if person else "Not found")
    print("When is the performance happening?:", date[0] if date else "Not found")
    print("Where will the concert take place?:", place[0] if place else "Not found")
    print("Which company is releasing the album?:", company[0] if company else "Not found")
    print()
 

if __name__ == "__main__":
    main()
