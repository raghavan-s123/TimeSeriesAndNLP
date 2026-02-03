import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span


def main():

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Install using: python -m spacy download en_core_web_sm")
        sys.exit(1)

    text = "The dollar has hit its highest level against the euro after the Federal Reserve head said the US trade deficit is set to stabilize."
    doc = nlp(text)

    print("\n=== 7.3.1 NLP Processed Tokens ===")
    for token in doc:
        print(token.text, token.pos_, token.dep_)
    print()

    ner_entities = [(ent.text, ent.label_) for ent in doc.ents]

    print("=== 7.3.2 Default NER Output ===")
    ner_df = pd.DataFrame(ner_entities, columns=["Entity", "Label"])
    print(ner_df)
    print()

    matcher_single = PhraseMatcher(nlp.vocab, attr="LOWER")

    currency_terms = ["the dollar", "the euro"]
    patterns = [nlp.make_doc(term) for term in currency_terms]
    matcher_single.add("CURRENCY", patterns)

    matches = matcher_single(doc)

    single_label_ents = list(doc.ents)
    for match_id, start, end in matches:
        span = Span(doc, start, end, label="CURRENCY")
        single_label_ents.append(span)

    doc_single = doc.copy()
    doc_single.ents = single_label_ents

    single_output = [(ent.text, ent.label_) for ent in doc_single.ents]

    print("=== 7.3.3 Single-Label Rule-based Matching Output ===")
    single_df = pd.DataFrame(single_output, columns=["Entity", "Label"])
    print(single_df)
    print()

    matcher_multi = PhraseMatcher(nlp.vocab, attr="LOWER")

    custom_phrases = {
        "CURRENCY": ["the dollar", "the euro"],
        "ECONOMIC_TERM": ["trade deficit"]
    }

    for label, phrases in custom_phrases.items():
        patterns = [nlp.make_doc(phrase) for phrase in phrases]
        matcher_multi.add(label, patterns)

    matches = matcher_multi(doc)

    multi_label_ents = list(doc.ents)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = Span(doc, start, end, label=label)
        multi_label_ents.append(span)

    unique_ents = []
    seen_tokens = set()


    for ent in sorted(multi_label_ents, key=lambda e: (e.start, -e.end)):
        if all(token.i not in seen_tokens for token in ent):
            unique_ents.append(ent)
            seen_tokens.update(token.i for token in ent)

    doc_multi = doc.copy()
    doc_multi.ents = unique_ents

    multi_output = [(ent.text, ent.label_) for ent in doc_multi.ents]

    print("=== 7.3.4 Multi-Label Rule-based Matching Output ===")
    multi_df = pd.DataFrame(multi_output, columns=["Entity", "Label"])
    print(multi_df)
    print()

    print("=== 7.3.5 Comparison Summary ===")

    print("\nDefault NER Entities:")
    print(ner_df)

    print("\nNER + Single-Label Rule-based Entities:")
    print(single_df)

    print("\nNER + Multi-Label Rule-based Entities:")
    print(multi_df)

    print("\nObservation:")
    print("- Default NER may miss domain-specific entities like currencies.")
    print("- Rule-based matching accurately captures predefined terms.")
    print("- Multi-label matching provides richer, task-specific entity extraction.")


if __name__ == "__main__":
    main()