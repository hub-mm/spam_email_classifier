# ./scripts/data_labelling.py
import pandas as pd
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def is_spam(email_content):
    if not isinstance(email_content, str):
        return 0

    spam_keywords = [
        'free', 'winner', 'credit', 'offer', 'click here',
        'buy now', 'limited time', 'act now', 'congratulations',
        'unsubscribe', 'advertisement', 'ad', 'ads'
    ]
    pattern = re.compile(r'\b(' + '|'.join(spam_keywords) + r')\b', re.IGNORECASE)

    matches = pattern.findall(email_content)
    score = len(matches)

    return 1 if score >= 2 else 0


def label_emails(input_data, output_data):
    try:
        df = pd.read_csv(input_data)
        logging.info(f"Loaded {len(df)} emails from {input_data}")
    except FileNotFoundError:
        logging.error(f"Input file {input_data} not found.")
        return
    except pd.errors.EmptyDataError:
        logging.error(f"Input file {input_data} is empty.")
        return
    except Exception as e:
        logging.error(f"Error reading {input_data}: {e}")
        return

    logging.info('Labelling emails...')

    # Replace NaN with empty strings and ensure all entries are strings
    if 'email' not in df.columns:
        logging.error("'email' column not found in the input CSV.")
        return

    df['email'] = df['email'].fillna('').astype(str)
    logging.info(f"Total emails to label: {len(df)}")

    # Apply spam labeling
    df['label'] = df['email'].apply(is_spam)

    try:
        df.to_csv(output_data, index=False)
        logging.info(f"Labeled emails saved to {output_data}")
    except Exception as e:
        logging.error(f"Error saving to {output_data}: {e}")


if __name__ == '__main__':
    input_csv = './data/processed/emails.csv'
    output_csv = './data/processed/email_labelled.csv'

    label_emails(input_csv, output_csv)