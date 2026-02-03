import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
import spacy

warnings.simplefilter(action='ignore')

def main():
    filename = input().strip()
    
    file_path = os.path.join(sys.path[0], filename)
    
    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    try:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found.")
            print("Run: python -m spacy download en_core_web_sm")
            sys.exit(1)

        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        lines = full_content.splitlines()
        print("First 10 lines from the file:")
        for line in lines[:10]:
            print(line)
        print() 
        
        doc = nlp(full_content)
        
      
        tokens_list = [token.text for token in doc[:20]]
        print("First 20 tokens:")
        print(tokens_list)
        print()
        
        print("POS Tagging Output:")
        print("Word\tPOS\tTag")
        print("-" * 30)
        
        for token in doc:
            print(f"{token.text}\t{token.pos_}\t{token.tag_}")

    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
