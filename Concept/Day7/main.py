import spacy
import pandas as pd
from spacy import displacy
import os
import webbrowser

# ================================
# LOAD NLP MODEL
# ================================
nlp = spacy.load("en_core_web_sm")

# ================================
# SAMPLE SENTENCES
# ================================
sentences = [
    "The dollar has hit its highest level against the euro after the Federal Reserve head said the US trade deficit is set to stabilize.",
    "Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn.",
    "The cat wearing sunglasses danced on the rooftop under the moonlight."
]

# ================================
# FUNCTION: DEPENDENCY TABLE
# ================================
def dependency_table(doc):
    data = []
    for token in doc:
        data.append([
            token.text,
            token.lemma_,
            token.pos_,
            token.dep_,
            token.head.text
        ])

    df = pd.DataFrame(data, columns=[
        "Token",
        "Lemma",
        "POS",
        "Dependency",
        "Head Word"
    ])

    return df

# ================================
# FUNCTION: VISUALIZE TREE IN BROWSER
# ================================
def visualize_tree(doc, filename="dependency_tree.html"):
    html = displacy.render(
        doc,
        style="dep",
        options={"distance": 110}
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open("file://" + os.path.realpath(filename))


# ================================
# PROCESS SAMPLE SENTENCES
# ================================
print("\n================ SAMPLE SENTENCES ================\n")

for i, sent in enumerate(sentences):
    print("\n" + "=" * 90)
    print("Sentence:\n", sent)

    doc = nlp(sent)

    df = dependency_table(doc)
    print("\nDependency Table:\n")
    print(df)

    print("\nDependency Tree opening in browser...\n")
    visualize_tree(doc, f"sample_tree_{i}.html")


# ================================
# PROCESS NEWS DATASET FILE
# ================================
print("\n================ NEWS DATASET ANALYSIS ================\n")

file_path = "Resources/ML471_S7_Datafile_Concept.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

# Only show first 5 sentences visually (avoid too many tabs)
for i, sent in enumerate(doc.sents):
    print("\n" + "-" * 90)
    print(f"Sentence {i + 1}:\n", sent.text.strip())

    df = dependency_table(sent)
    print("\nDependency Table:\n")
    print(df)

    if i < 5:
        print("\nDependency Tree opening in browser...\n")
        visualize_tree(sent, f"news_tree_{i}.html")

# ================================
# SAVE ALL DEPENDENCIES TO CSV
# ================================
all_data = []

for sent in doc.sents:
    for token in sent:
        all_data.append([
            sent.text,
            token.text,
            token.pos_,
            token.dep_,
            token.head.text
        ])

final_df = pd.DataFrame(all_data, columns=[
    "Sentence",
    "Token",
    "POS",
    "Dependency",
    "Head Word"
])

final_df.to_csv("dependency_output.csv", index=False)

print("\n✅ Dependency analysis completed.")
print("📁 Trees saved as HTML files in project folder.")
print("📊 dependency_output.csv created.")