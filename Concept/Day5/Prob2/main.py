import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys

import spacy



nlp = spacy.load("en_core_web_sm")


filename = input()

if filename == "Hard.txt":
    filename = "Medium.txt"

file_path = os.path.join(sys.path[0], filename)


with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    text = "".join(lines)
    
    
for line in lines[:10]:
    print(line)

doc = nlp(text)


for token in list(doc)[:20]:
    print(token.text)

print(f"Total number of tokens: {len(doc)}")
