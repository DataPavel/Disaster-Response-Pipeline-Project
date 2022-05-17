# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    This function loads data from csv files and merges the filepaths

    INPUT: string - file path
    OUTPUT: DataFrame - merged dataframe

    """


    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how = 'left')

    return df

def clean_data(df):
    """
    This function wrangles the data and returns clean dataframe

    INPUT: DataFrame - loaded dataframe
    OUTPUT: DataFrame - clean dataframe
    """

    # Split category variable into separate columns
    cats = df.iloc[:,-1].str.split(';', expand = True)

    # Add column names to the resulting dataframe
    column_names = []
    for i in cats[:1].values.flatten().tolist():
        column_names.append(i.replace('-1', '').replace('-0', ''))
    cats.columns = column_names

    # Convert category values to just numbers 0 or 1
    for column in cats:
        cats[column] = cats[column].str[-1].astype(int)
    # Concatenate dataframes
    df = pd.concat([df.drop('categories', axis=1), cats], axis=1)
    # Replace "2" by "1" in the related variable
    df.related = np.where(df.related >1, 1, df.related)
    # Drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    """
    This function saves data into database

    INPUT: DataFrame, database_filename
    OUTPUT: None

    """
    conn = sqlite3.connect(database_filename)
    df.to_sql('disaster', conn, index=False, if_exists='replace')


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
