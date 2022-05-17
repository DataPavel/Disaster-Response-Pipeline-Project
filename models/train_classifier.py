import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

def load_data(database_filepath):
    """
    This function loads data and divide it by features and labels

    INPUT: database path
    OUTPUT: Labels, targets, list of target column_names
    """
    # Load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("""
    SELECT *
    FROM disaster
    """, con = conn)
    # Divide dataframe by features and labels
    X = df['message'].values
    y = df.iloc[:,4:]
    categories = list(y.columns)

    return X, y, categories


def tokenize(text):
    """
    This function converts strings into clean tokens

    INPUT: string - text
    OUTPUT: clean tokens
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    This function builds ML model

    INPUT: None
    OUTPUT: model
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model disploying classification report

    INPUT: ML model, test labels, test targets, list of categories
    OUTPUT: None
    """
    y_pred = model.predict(X_test)

    for i, b in enumerate(category_names):
        print(b)
        print(classification_report(Y_test[b], y_pred[:,i], labels =[1,0],
         zero_division = 0))




def save_model(model, model_filepath):
    """
    This function saves model into pickle file

    INPUT: ML model, path where to save the pickle file
    OUTPUT: None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
