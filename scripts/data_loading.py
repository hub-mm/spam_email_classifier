# ./scripts/data_loading.py
import os
import pandas as pd
from email import policy
from email.parser import BytesParser


def load_emails(dataset_path, max_users=None, max_emails_per_user=None):
    emails = []
    user_count = 0

    # Iterate over each user directory in the dataset_path
    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)

        if not os.path.isdir(user_path):
            # Skip if not a directory
            continue

        if max_users and user_count >= max_users:
            break

        email_count = 0

        # Walk through all subdirectories and files within the user directory
        for root, dirs, files in os.walk(user_path):
            for email_file in files:
                if max_emails_per_user and email_count >= max_emails_per_user:
                    break

                file_path = os.path.join(root, email_file)

                # Ensure the path is a file before processing
                if not os.path.isfile(file_path):
                    continue

                try:
                    with open(file_path, 'rb') as f:
                        msg = BytesParser(policy=policy.default).parse(f)

                    if msg.is_multipart():
                        body = ''
                        for part in msg.iter_parts():
                            if part.get_content_type() == 'text/plain':
                                body += part.get_content()
                    else:
                        body = msg.get_content()

                    emails.append(body)
                    email_count += 1

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

        user_count += 1

    return emails

def save_emails_to_csv(emails, output_path):
    df = pd.DataFrame({'email': emails})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(emails)} emails to {output_path}")

if __name__ == '__main__':
    raw_data_path = './data/raw/maildir'
    output_csv = './data/processed/emails.csv'

    MAX_USER = 5
    MAX_EMAILS_PER_USER = None

    print('Loading emails...')
    data = load_emails(raw_data_path, max_users=MAX_USER, max_emails_per_user=MAX_EMAILS_PER_USER)
    print(f"Total emails loaded: {len(data)}")

    print('Saving emails to CSV...')
    save_emails_to_csv(data, output_csv)