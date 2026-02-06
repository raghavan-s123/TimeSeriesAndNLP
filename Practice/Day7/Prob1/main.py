import os
import sys
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:
    print("SpaCy library not found.")
    sys.exit(1)

def main():
    file_name = input("Enter text file name: ")
    
    file_path = os.path.join(sys.path[0], file_name)

    if not os.path.exists(file_path):
        print(f"Error: File '{file_name}' not found.")
        sys.exit(1)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Install it using: python -m spacy download en_core_web_sm")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print("=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    doc = nlp(content)

    athlete_matcher = PhraseMatcher(nlp.vocab)
    athletes = ["Sarah Claxton", "Sonia O'Sullivan", "Irina Shevchenko"]
    athlete_patterns = [nlp.make_doc(text) for text in athletes]
    athlete_matcher.add("AthleteList", athlete_patterns)

    athlete_matches = athlete_matcher(doc)
    print("=== Matched Athlete Names ===")
    if not athlete_matches:
        print("No athlete names found.")
    else:
        for match_id, start, end in athlete_matches:
            span = doc[start:end]
            print(f"- {span.text}")
    print()

    # 3. Extract Sports Events
    event_matcher = PhraseMatcher(nlp.vocab)
    events = [
        "European Indoor Championships", 
        "World Cross Country Championships", 
        "London marathon", 
        "Bupa Great Ireland Run"
    ]
    event_patterns = [nlp.make_doc(text) for text in events]
    event_matcher.add("EventList", event_patterns)

    event_matches = event_matcher(doc)
    print("=== Matched Sports Events ===")
    if not event_matches:
        print("No sports events found.")
    else:
        for match_id, start, end in event_matches:
            span = doc[start:end]
            print(f"- {span.text}")

if __name__ == "__main__":
    main()
