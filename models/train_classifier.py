# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.metrics import classification_report
import pickle
import sys


def load_data(database_filepath):
    '''

    Parameters
    ----------
    database_filepath : path of the sqlite database to load data from

    Returns
    -------
    X : messages texts that are used as input feature to the classifier
    Y : messages categories 
    category_names : names of messages categories

    '''
    # load data from database
    engine = create_engine('sqlite:///%s')%database_filepath
    df = pd.read_sql_table("messages_categories", engine)
    # define features and labels
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''

    Parameters
    ----------
    text : strint text to be tokenized
    
    Returns
    -------
    clean_tokens : list of tokens found in the input text

    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # iterate through each token
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    
    Returns
    -------
    model_pipeline : message categories classification pipeline including grid search with multiple parameters to try
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier',MultiOutputClassifier(RandomForestClassifier()))
    ])
    # define parameters for GridSearchCV
    gs_parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'classifier__criterion' :['gini', 'entropy'],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__min_samples_split': [2, 3, 4],

    }
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=gs_parameters)

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''

    Parameters
    ----------
    model : ML model to be evaluated
    X_test : input features to test the model
    Y_test : ground truth labels for the test data
    category_names : names of messages categories

    Returns
    -------
    None.

    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names.values))



def save_model(model, model_filepath):
    '''

    Parameters
    ----------
    model : ML model to be saved
    model_filepath : required model file path to save in

    Returns
    -------
    None.

    '''
    pickle.dump(model, open(model_filepath, "wb"))



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