# ./scripts/deploy_model.py
from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, 'models', 'spam_classifier.pkl')
vectorizer_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logging.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Model or vectorizer file not found: {e}")
    raise e
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    raise e

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        return ''


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email_content = request.form.get('email', '')
        if not email_content:
            error_message = "Please enter the email content."
            return render_template('index.html', error=error_message)

        preprocessed = preprocess_text(email_content)
        if not preprocessed:
            error_message = "Error in preprocessing the email content."
            return render_template('index.html', error=error_message)

        features = vectorizer.transform([preprocessed])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            'spam': bool(prediction),
            'probability': round(float(probability), 2)
        }

        logging.info(
            f"Email classified as {'Spam' if result['spam'] else 'Not Spam'} with probability {result['probability']}")

        return render_template('index.html', result=result, email=email_content)

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_content = data.get('email', '')
    if not email_content:
        return jsonify({'error': 'No email content provided'}), 400

    preprocessed = preprocess_text(email_content)
    if not preprocessed:
        return jsonify({'error': 'Error in preprocessing the email content'}), 500

    features = vectorizer.transform([preprocessed])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = {
        'spam': bool(prediction),
        'probability': round(float(probability), 2)
    }

    logging.info(
        f"API: Email classified as {'Spam' if result['spam'] else 'Not Spam'} with probability {result['probability']}")

    return jsonify(result)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    # For development purposes only. Use Gunicorn for production.
    app.run(debug=True)