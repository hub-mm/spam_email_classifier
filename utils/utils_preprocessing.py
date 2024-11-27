# ./utils/utils_preprocessing.py
import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text, flags=re.MULTILINE)
        # Remove numbers and punctuation
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize
        words = nltk.word_tokenize(text)
        # Lemmatize without removing stopwords
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")

        return ''