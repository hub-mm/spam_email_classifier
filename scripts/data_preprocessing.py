# ./scripts/data_preprocessing.py
import pandas as pd
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from utils.utils_preprocessing import preprocess_text

def preprocess_emails(input_data, output_data):
    try:
        df = pd.read_csv(input_data)
        logging.info('Loaded emails for preprocessing...')
    except FileNotFoundError:
        logging.error(f"Input file {input_data} not found.")
        return
    except Exception as e:
        logging.error(f"Error reading {input_data}: {e}")
        return

    logging.info('Preprocessing emails...')

    df['email'] = df['email'].fillna('').astype(str)
    df['email_clean'] = df['email'].apply(preprocess_text)

    try:
        df.to_csv(output_data, index=False)
        logging.info(f"Preprocessed emails saved to {output_data}")
    except Exception as e:
        logging.error(f"Error saving to {output_data}: {e}")

if __name__ == '__main__':
    input_csv = './data/processed/emails_labelled.csv'
    output_csv = './data/processed/emails_preprocessed.csv'

    preprocess_emails(input_csv, output_csv)