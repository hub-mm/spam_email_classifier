# ./scripts/extract_features.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def extract_tfidf_features(corpus):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    x = vectorizer.fit_transform(corpus)
    return x, vectorizer


def extract_features(input_csv, x_output, y_output, vectorizer_output):
    try:
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} labeled emails from {input_csv}")
    except FileNotFoundError:
        logging.error(f"Input file {input_csv} not found.")
        return
    except pd.errors.EmptyDataError:
        logging.error(f"Input file {input_csv} is empty.")
        return
    except Exception as e:
        logging.error(f"Error reading {input_csv}: {e}")
        return

    if 'email' not in df.columns or 'label' not in df.columns:
        logging.error("Required columns ('email', 'label') not found in the input CSV.")
        return

    # Replace NaN with empty strings and ensure all entries are strings
    df['email'] = df['email'].fillna('').astype(str)
    logging.info("Replaced NaN values with empty strings in 'email' column.")

    corpus = df['email'].tolist()
    labels = df['label'].tolist()

    logging.info("Extracting TF-IDF features...")
    try:
        x, vectorizer = extract_tfidf_features(corpus)
        logging.info("TF-IDF feature extraction completed.")
    except ValueError as ve:
        logging.error(f"ValueError during TF-IDF extraction: {ve}")
        return
    except Exception as e:
        logging.error(f"Unexpected error during TF-IDF extraction: {e}")
        return

    # Save features and labels
    try:
        # Save TF-IDF features as a sparse matrix using joblib
        joblib.dump(x, x_output)
        logging.info(f"TF-IDF features saved to {x_output}")

        # Save labels
        pd.Series(labels).to_csv(y_output, index=False, header=['label'])
        logging.info(f"Labels saved to {y_output}")

        # Save the vectorizer for future use
        joblib.dump(vectorizer, vectorizer_output)
        logging.info(f"Vectorizer saved to {vectorizer_output}")

    except Exception as e:
        logging.error(f"Error saving outputs: {e}")


if __name__ == '__main__':
    input_csv = './data/processed/emails_preprocessed.csv'
    x_output = './models/X.pkl'
    y_output = './models/y.csv'
    vectorizer_output = './models/tfidf_vectorizer.pkl'

    extract_features(input_csv, x_output, y_output, vectorizer_output)