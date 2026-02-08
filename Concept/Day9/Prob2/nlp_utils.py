import re
import string


# ============================
# Function 1: Clean Text
# ============================
def clean_text(text):
    """
    Cleans input text by:
    - Converting to lowercase
    - Removing URLs
    - Removing numbers
    - Removing punctuation
    - Removing extra spaces
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================
# Function 2: Split Emotion Labels
# ============================
def split_labels(label_text):
    """
    Converts emotion labels from string to list.
    Example:
        "joy,sadness,anger" -> ['joy', 'sadness', 'anger']
    """
    if not isinstance(label_text, str):
        return []

    return [label.strip() for label in label_text.split(',') if label.strip()]
