# ./scripts/data_preprocessing.py
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URlS
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    text = re.sub(r'\s+', '', text).strip()
    # Tokenise
    words = text.split()
    # Remove stopwords and lemmatize
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop]

    return ''.join(words)

def preprocess_emails(input_data, output_data):
    df = pd.read_csv(input_data)
    print('Preprocessing emails...')

    df['email'] = df['email'].fillna('').astype(str)

    df['email_clean'] = df['email'].apply(preprocess_text)
    df.to_csv(output_data, index=False)
    print(f"Preprocessed emails saved to {output_data}")

if __name__ == '__main__':
    input_csv = './data/processed/emails_labelled.csv'
    output_csv = './data/processed/emails_preprocessed.csv'

    preprocess_emails(input_csv, output_csv)