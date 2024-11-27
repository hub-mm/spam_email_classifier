# ./scripts/deploy_model.py
from flask import Flask, request, jsonify, render_template
import joblib
import logging
import os
from markupsafe import escape
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.utils_preprocessing import preprocess_text

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secure_secret_key')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

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

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email_content = request.form.get('email', '')
        if not email_content:
            error_message = "Please enter the email content."
            return render_template('index.html', error=error_message)
        try:
            preprocessed = preprocess_text(email_content)
            if not preprocessed.strip():
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
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            error_message = "An error occurred during prediction. Please try again."
            return render_template('index.html', error=error_message)

        # Use escape to prevent XSS when rendering user input
        return render_template('index.html', result=result, email=escape(email_content))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_content = data.get('email', '')
    if not email_content:
        return jsonify({'error': 'No email content provided'}), 400
    try:
        preprocessed = preprocess_text(email_content)
        if not preprocessed.strip():
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
    except Exception as e:
        logging.error(f"Error during API prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

    return jsonify(result)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=False)