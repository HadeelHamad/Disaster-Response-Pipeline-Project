# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Parameters
    ----------
    messages_filepath : path for csv messages file
    categories_filepath : path for csv categories file

    Returns
    -------
    df : merged dataframe of messages and categories datasets
    '''
  
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on='id')
    return df




def clean_data(df):
    '''

    Parameters
    ----------
    df : merged dataframe
    Returns
    -------
    df : cleaned dataframe, after converting 'categories' column to list of columns of categories values 
    '''
    ##1.Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[0:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    ##2.Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # first I assumed that all coulmns values are 1 or 0, then I completed building the pipeline but an erreor occured when evaluating the
    # model that there is a column category has 3 values (0,1,2) so I added the folowing line to remove 2 values
    categories= categories.replace(2, 1)

    ##3.Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    ##4.Remove duplicates
    df = df.drop_duplicates()
    return df
        
def save_data(df, database_filename):
    '''

    Parameters
    ----------
    df : cleaned dataframe ready to be saved to sql database
    database_filename : database file name 

    Returns
    -------
    None.

    '''
    #Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///%s'%database_filename)
    df.to_sql('messages_categories', engine, index=False)
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()