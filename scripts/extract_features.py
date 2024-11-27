# ./scripts/extract_features.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def extract_features(input_csv, x_train_output, x_test_output, y_train_output, y_test_output, vectorizer_output):
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

    if 'email_clean' not in df.columns or 'label' not in df.columns:
        logging.error("Required columns ('email_clean', 'label') not found in the input CSV.")
        return

    # Replace NaN with empty strings and ensure all entries are strings
    df['email_clean'] = df['email_clean'].fillna('').astype(str)
    logging.info("Replaced NaN values with empty strings in 'email_clean' column.")

    corpus = df['email_clean'].tolist()
    labels = df['label'].tolist()

    # Split data into training and test sets
    logging.info('Splitting data into train and test sets...')
    x_train_texts, x_test_texts, y_train, y_test = train_test_split(
        corpus, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logging.info(f"Training samples: {len(x_train_texts)}, Testing samples: {len(x_test_texts)}")

    # Fit vectorizer on training data
    logging.info("Extracting TF-IDF features...")
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 3)
        )
        x_train = vectorizer.fit_transform(x_train_texts)
        x_test = vectorizer.transform(x_test_texts)
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
        joblib.dump(x_train, x_train_output)
        joblib.dump(x_test, x_test_output)
        logging.info(f"TF-IDF training features saved to {x_train_output}")
        logging.info(f"TF-IDF testing features saved to {x_test_output}")

        # Save labels
        pd.Series(y_train).to_csv(y_train_output, index=False, header=['label'])
        pd.Series(y_test).to_csv(y_test_output, index=False, header=['label'])
        logging.info(f"Training labels saved to {y_train_output}")
        logging.info(f"Testing labels saved to {y_test_output}")

        # Save the vectorizer for future use
        joblib.dump(vectorizer, vectorizer_output)
        logging.info(f"Vectorizer saved to {vectorizer_output}")

    except Exception as e:
        logging.error(f"Error saving outputs: {e}")


if __name__ == '__main__':
    input_csv = './data/processed/emails_preprocessed.csv'
    x_train_output = './models/X_train.pkl'
    x_test_output = './models/X_test.pkl'
    y_train_output = './models/y_train.csv'
    y_test_output = './models/y_test.csv'
    vectorizer_output = './models/tfidf_vectorizer.pkl'

    extract_features(
        input_csv,
        x_train_output,
        x_test_output,
        y_train_output,
        y_test_output,
        vectorizer_output
    )