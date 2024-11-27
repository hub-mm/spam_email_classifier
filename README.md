# Spam Email Classifier using Machine Learning

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Deployment](#deployment)
- [Web Application](#web-application)
- [API Endpoint](#api-endpoint)
- [Contributions](#contributions)
- [Acknowledgment](#acknowledgement)

## Overview
This project implements a machine learning pipeline to classify emails as spam or not spam using the Enron email dataset.
The pipeline includes data loading, preprocessing, feature extraction, model training, evaluation, and deployment as a web application.

## Dataset
The [Enron email](https://www.cs.cmu.edu/~enron/) dataset is a large corpus of real emails that have been made public and are suitable for research purposes.
It contains approximately 0.5 million emails from about 150 users.

## Project Structure


## Installation
### Prerequisites
- Python 3.7 or higher
- [Git](https://git-scm.com)

### Steps
1. **Clone Repository**
    ```bash 
    git clone
    ```
2. **Move into Repository**
    ```bash
   cd spam_email_classifier
   ```
3. **Create Virtual Environment**
    ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows use:
   venv\Scripts\activate
   ```
4. **Install Required Packages**
    ```bash
   pip install -r requirements.txt
   ```
5. **Download NLTK data**
    ```bash
   python utils/setup_nltk.py
   ```
6. **Download the Enron Dataset**  
   Download the dataset from [here](https://www.cs.cmu.edu/~enron/) and extract it into 
   ***data/processed/raw/maildir/***

## Usage
### Data Preparation
1. **Load Emails**
    ```bash
   python scripts/data_loading.py
   ```
This script loads emails from the Enron dataset and saves them to
***data/processed/emails.csv***


2. **Label Emails**
    ```bash
   python scripts/data_labelling.py
   ```
Labels emails as spam or not spam based on keyword matching and saves the result to 
***data/processed/emails_labelled.csv***


3. **Preprocess Emails**
    ```bash
   python scripts/data_preprocessing.py
   ```
Cleans the email text and saves the preprocessed data to
***data/processed/emails_preprocessed.csv***

### Model Training
1. **Extract Features**
    ```bash
   python scripts/extract_features.py
   ```
Extracts TF-IDF features from the preprocessed emails and splits the data into training and testing sets.

2. **Train Model**
    ```bash
   python scripts/train_model.py
   ```
Trains a Multinomial Naive Bayes classifier and saves the model to
***models/spam_classifier.pkl***

### Model Evaluation
```bash
    python scripts/evaluate_model.py
```
Evaluates the trained model on the test set and displays classification metrics and plots.

### Deployment
```bash
    python scripts/deploy_model.py
```
Runs the Flask web application for classifying emails via a web interface or API.


## Web Application
The web application allows users to input email content and classify it as spam or not spam.
- **Home Page**
- ** Classification Result

### Running the App
```bash
    python scripts/deploy_model.py
```
Navigate to [http://localhost:5000](http://localhost:5000) in your web browser.

## API Endpoint
### /predict (POST)
- **Description:** Classifies the given email content.
- **Requested Body:** { 'email': 'Your email content here' }
- **Response:** { 'spam': true, 'probability': 0.95 }

## Contributions
Contributions are welcome! Please open an issue or submit a pull request for any bugs, enhancements, or features.

## Acknowledgement
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org)
- [Flask](https://flask.palletsprojects.com/en/stable/)