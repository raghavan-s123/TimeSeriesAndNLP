import os
import sys
import warnings
import spacy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore')

filename = input()

if not filename.lower().endswith('.csv'):
    print(f"Error: File '{filename}' must have a .csv extension.")
    sys.exit(1)

file_path = os.path.join(sys.path[0], filename)

if not os.path.exists(file_path):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

nlp = spacy.load("en_core_web_sm")

with open(file_path, 'r', encoding='utf-8') as f:
    raw_content = f.read()
    lines = raw_content.splitlines()

print("First 10 lines from the file:")
for line in lines[:10]:
    print(line)
print() 

doc = nlp(raw_content)
tokens = [token.text for token in doc]

print("First 20 tokens:")
print(tokens[:20])
