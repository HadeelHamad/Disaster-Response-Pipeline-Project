# Disaster Response Pipeline Project
### Project Overview:
In this project, data engineering techniques are applied to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages to 36 defined disaster's categories so that the message can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Files Description:

1. ETL Pipeline
In a Python script, process_data.py, data cleaning pipeline will do the following:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, machine learning pipeline will do the following:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
In a Python script, app/run.py, a web app will do the following:
- Enables an emergency worker to input a message and get its category
- Shows data visualisations for the training data.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

