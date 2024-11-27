# train_model.py
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_data(x_train_path, y_train_path):
    x_train = joblib.load(x_train_path)
    y_train_df = pd.read_csv(y_train_path)
    y_train = y_train_df['label'].values.ravel()
    return x_train, y_train


def train_model(x_train, y_train):
    logging.info('Training MultinomialNB model with hyperparameter tuning...')
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
    grid_search.fit(x_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    logging.info(f"Best alpha found: {best_alpha}")

    model = MultinomialNB(alpha=best_alpha)
    model.fit(x_train, y_train)

    return model


def save_model(model, output_path):
    joblib.dump(model, output_path)
    logging.info(f"Trained model saved to {output_path}")


def main():
    x_train_path = './models/X_train.pkl'
    y_train_path = './models/y_train.csv'
    model_output = './models/spam_classifier.pkl'

    x_train, y_train = load_data(x_train_path, y_train_path)

    model = train_model(x_train, y_train)
    save_model(model, model_output)


if __name__ == '__main__':
    main()