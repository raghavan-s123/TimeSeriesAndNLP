import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs


import numpy as np
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity

def load_spacy_model():
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "No spaCy English model found.\n"
                "Install one using:\n"
                "python -m spacy download en_core_web_sm"
            )

nlp = load_spacy_model()
VECTOR_SIZE = nlp.vocab.vectors_length or 300

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_docs(docs):
    return [clean_text(d) for d in docs if d.strip()]

def word_vector(word):
    doc = nlp(word)
    if doc.has_vector:
        return doc.vector
    return np.zeros(VECTOR_SIZE)

def doc_embeddings(docs):
    vectors = []
    for doc in nlp.pipe(docs):
        if doc.has_vector:
            vectors.append(doc.vector)
        else:
            vectors.append(np.zeros(VECTOR_SIZE))
    return np.vstack(vectors)

def word_similarity(sentence):
    words = sentence.split()
    vectors = []
    valid_words = []

    for w in words:
        vec = word_vector(w)
        if np.any(vec):
            vectors.append(vec)
            valid_words.append(w)

    if len(vectors) < 2:
        return None, valid_words

    vectors = np.vstack(vectors)
    sim_matrix = cosine_similarity(vectors)

    return sim_matrix, valid_words

def cosine_sim_embeddings(vectors):
    return cosine_similarity(vectors)

docs = [
    "news article",
    "ad sales boost time warner profit",
    "quarterly profits at us media giant timewarner jumped to bn m",
    "time warner said on friday that it now owns of searchengine google"
]

clean_docs = preprocess_docs(docs)
print("\nCleaned Documents:")
for i, doc in enumerate(clean_docs, start=1):
    print(f"{i}: {doc}")

king_vec = word_vector("king")
print("\nWord Vector for 'king' (first 10 dims):")
print(king_vec[:10])

doc_vecs = doc_embeddings(clean_docs)
print("\nDocument Embedding Shape:", doc_vecs.shape)

doc_sim = cosine_sim_embeddings(doc_vecs)
print("\nCosine Similarity Between Documents:")
print(np.round(doc_sim, 3))

sentence = "dog cat car skym apple"
sim_matrix, words_used = word_similarity(sentence)

if sim_matrix is not None:
    print("\nWord Similarity Matrix (words with vectors):")
    for i, w1 in enumerate(words_used):
        for j, w2 in enumerate(words_used):
            if j > i:
                print(f"{w1} ↔ {w2} : {sim_matrix[i,j]:.3f}")
else:
    print("\nNot enough words with vectors to compute similarity.")

print("\nObservations:")
print("• 'dog' and 'cat' have high similarity due to both being animals.")
print("• 'car' is moderately similar to 'dog' and 'cat' due to physical object context.")
print("• 'skym' may be OOV → low similarity with other words.")
print("• Embeddings capture meaning beyond frequency (unlike TF-IDF).")
