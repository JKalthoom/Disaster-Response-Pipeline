import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load data from the SQLite database
    
    Args:
    database_filepath: string. Filepath for the database.
    
    Returns:
    X: DataFrame. Features.
    Y: DataFrame. Labels.
    category_names: list of strings. Category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text
    
    Args:
    text: string. Text to be tokenized.
    
    Returns:
    clean_tokens: list of strings. Tokenized text.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Build machine learning pipeline and perform grid search
    
    Returns:
    cv: GridSearchCV object. Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model
    
    Args:
    model: model object. Trained model.
    X_test: DataFrame. Test features.
    Y_test: DataFrame. Test labels.
    category_names: list of strings. Category names.
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f'Category: {category_names[i]}')
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model to a pickle file
    
    Args:
    model: model object. Trained model.
    model_filepath: string. Filepath for the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()