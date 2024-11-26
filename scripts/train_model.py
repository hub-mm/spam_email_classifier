# train_model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(x_path, y_path):
    x =  joblib.load(x_path)

    y_df =  pd.read_csv(y_path)
    y = y_df['label'].values.ravel()

    return x, y

def train_model(x_train, y_train):
    # model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    print('Training Random Forest model...')

    model.fit(x_train, y_train)

    return model

def evaluate_model(model, x_test, y_test):
    print('Evaluating model...')

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(cm)

def save_model(model, output_path):
    joblib.dump(model, output_path)
    print(f"Trained model saved to {output_path}")

def main():
    x_path = './models/X.pkl'
    y_path = './models/y.csv'
    model_output = './models/spam_classifier.pkl'

    x, y = load_data(x_path, y_path)

    print('Splitting data into train and test sets...')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training sample: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model, model_output)

if __name__ == '__main__':
    main()