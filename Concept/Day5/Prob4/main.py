import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings

warnings.simplefilter(action='ignore')

import spacy
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')

if not spacy.util.is_package("en_core_web_sm"):
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

nlp = spacy.load("en_core_web_sm")

filename = input("Enter text file name for full text processing:")
print()

if not filename.lower().endswith('.txt'):
    sys.exit(1)

file_path = os.path.join(sys.path[0], filename)

if not os.path.exists(file_path):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    text_content = f.read()

print("Original Text Sample:")
print(text_content[:300])
print()

print("=== Lemmatization: Individual Words ===")
sample_words = "friendship studied was am is organizing matches".split()
for word in sample_words:
    lemma = nlp(word)[0].lemma_
    print(f"{word} -> {lemma}")
print()

print("=== Stemming: Individual Words ===")
for word in sample_words:
    stem = stemmer.stem(word)
    print(f"{word} --> {stem}")
print()

print("=== Lemmatization: Full Text ===")
doc = nlp(text_content)
full_tokens = [token for token in doc if not token.is_space]
for token in full_tokens[:50]:
    print(f"{token.text} --> {token.lemma_}")
print()

print("=== Stemming: Full Text ===")
for token in full_tokens[:50]:
    stem = stemmer.stem(token.text.lower())
    print(f"{token.text} --> {stem}")
print()

print("=== Practice 6.2: Lemmatization vs Stemming ===")
practice_words = ["running", "good", "universities", "flies", "fairer", "is"]
print("Word\t\tLemma\t\tStem")
print("-" * 42)
for word in practice_words:
    lemma = nlp(word)[0].lemma_
    stem = stemmer.stem(word)
    print(f"{word}\t\t{lemma}\t\t{stem}")
print()

print("Conclusion:")
print("Lemmatization produces dictionary-based meaningful root words, while stemming may distort words by chopping suffixes. For NLP tasks like search, topic modeling, and information retrieval, lemmatization gives better and cleaner output.")
