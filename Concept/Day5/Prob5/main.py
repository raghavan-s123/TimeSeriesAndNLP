#TESTCASES WONT PASS ONLY SAMPLE WILL PASS

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
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

  print("\n=== Original Text Sample (First 300 chars) ===")
  print(content[:300])
  print()

  try:
    nlp = spacy.load("en_core_web_sm")
  except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

  doc = nlp(content)
  filtered_words = [token.text for token in doc if not token.is_stop and not token.is_space]

  print("=== Text After Stop Word Removal (Sample) ===")
  print(" ".join(filtered_words[:50]))
  print()

  custom_stopwords = {"whatever", "set", "example", "whenever"}
  nlp.Defaults.stop_words |= custom_stopwords

  for word in custom_stopwords:
    nlp.vocab[word].is_stop = True

  print("Custom stop words added: {'set', 'example', 'whenever', 'whatever'}")
  print("Is 'example' a stop word?", nlp.vocab["example"].is_stop)
  print("Total Stop Words Now:", len(nlp.Defaults.stop_words))
  print()

  remove_words = {"example"}
  nlp.Defaults.stop_words -= remove_words
  nlp.vocab["example"].is_stop = False

  print("Removed stop word:", remove_words)
  print("Is 'example' a stop word now?", nlp.vocab["example"].is_stop)
  print("Total Stop Words After Removal:", len(nlp.Defaults.stop_words))
  print()

  def remove_stop_words(sentence):
    doc = nlp(sentence)
    return " ".join([token.text for token in doc if not token.is_stop])

  sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am learning Natural Language Processing with Python.",
    "What is the best way to learn natural language processing?"
  ]

  print("=== Stop Word Removal Examples ===")
  for i, s in enumerate(sentences, 1):
    print(f"\nInput {i}: {s}")
    print("After Stop Word Removal:", remove_stop_words(s))

if __name__ == "__main__":
  main()
