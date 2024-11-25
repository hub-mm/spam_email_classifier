# ./scripts/data_labeling.py
import pandas as pd
import re


def is_spam(email_content):
    if not isinstance(email_content, str):
        return 0

    spam_keyword = [
        'free', 'winner', 'credit', 'offer', 'click here',
        'buy now', 'limited time', 'act now', 'congratulations', 'unsubscribe'
    ]
    pattern = re.compile('|'.join(spam_keyword), re.IGNORECASE)

    return 1 if pattern.search(email_content) else 0

def label_emails(input_data, output_data):
    df = pd.read_csv(input_data)
    print('Labelling emails...')

    df['email'] = df['email'].fillna('').astype(str)
    
    df['label'] = df['email'].apply(is_spam)
    df.to_csv(output_data, index=False)
    print(f"Labeled emails saved to {output_data}")

if __name__ == '__main__':
    input_csv = './data/processed/emails.csv'
    output_csv = './data/processed/email_labelled.csv'

    label_emails(input_csv, output_csv)