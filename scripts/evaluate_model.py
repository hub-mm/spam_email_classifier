# ./scripts/evaluate_model.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def load_data(x_test_path, y_test_path):
    x_test = joblib.load(x_test_path)
    y_test_df = pd.read_csv(y_test_path)
    y_test = y_test_df['label'].values.ravel()
    return x_test, y_test


def evaluate(model, x_test, y_test):
    logging.info('Making predictions on test data...')
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    logging.info('Classification Report:')
    print(classification_report(y_test, y_pred))

    logging.info('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    auc = roc_auc_score(y_test, y_proba)
    logging.info(f"ROC-AUC Score: {auc:.2f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def main():
    model_path = './models/spam_classifier.pkl'
    x_test_path = './models/X_test.pkl'
    y_test_path = './models/y_test.csv'

    model = load_model(model_path)
    x_test, y_test = load_data(x_test_path, y_test_path)
    evaluate(model, x_test, y_test)


if __name__ == '__main__':
    main()