import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings

warnings.simplefilter(action='ignore')

try:
    import spacy
    from nltk.stem.snowball import SnowballStemmer
except ImportError:
    print("Required libraries missing. Install them using: pip install spacy nltk")
    sys.exit(1)

def run_pipeline():
    
    filename = input("Enter text file name: ")
    print()
    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
  

    print("Original Text Sample:")
    print(content[:300])
    print()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    doc = nlp(content)
    tokens = [token for token in doc if not token.is_space]
    
    print(f"Total Tokens Count: {len(tokens)}")
    print()

    lemmas = [token.lemma_ for token in tokens]
    
    print("=== Lemmatized Sample (First 20 tokens) ===")
    print(lemmas[:20])
    print()

    print("Word --> Lemma")
    for token, lemma in zip(tokens[:30], lemmas[:30]):
        print(f"{token.text} --> {lemma}")
    print()

    stemmer = SnowballStemmer(language='english')
    stems = [stemmer.stem(token.text.lower()) for token in tokens]

    print("=== Stemmed Sample (First 20 tokens) ===")
    print(stems[:20])
    print()

    print("Word --> Stem")
    for token, stem in zip(tokens[:30], stems[:30]):
        print(f"{token.text} --> {stem}")
    print()

    print("=== Comparison: Lemmatization vs Stemming ===")
    print("Word\t\tLemma\t\tStem")
    print("-" * 42)
    for t, l, s in zip(tokens[:30], lemmas[:30], stems[:30]):
        print(f"{t.text}\t\t{l}\t\t{s}")
    print()

    print("Conclusion:")
    print("Lemmatization produces dictionary-based meaningful root words, while stemming may distort words by chopping suffixes. For NLP tasks like search, topic modeling, and information retrieval, lemmatization gives better and cleaner output.")
    
if __name__ == "__main__":
    run_pipeline()
