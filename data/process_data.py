import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files
    
    Args:
    messages_filepath: string. Filepath for the messages CSV file.
    categories_filepath: string. Filepath for the categories CSV file.
    
    Returns:
    df: DataFrame. Merged data from messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataframe
    
    Args:
    df: DataFrame. Merged data from messages and categories.
    
    Returns:
    df: DataFrame. Cleaned data.
    """
    # Check for duplicates and drop them
    df = df.drop_duplicates(subset='id')

    # Split the categories column into separate columns
    categories_split = df['categories'].str.split(';', expand=True)
    
    # Use the first row to create column names
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories_split.columns = category_colnames

    # Convert category values to 0 or 1
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1].astype(int)
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_split], axis=1)

    # Check for duplicates in the final dataframe and drop them
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database
    
    Args:
    df: DataFrame. Cleaned data.
    database_filename: string. Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, if_exists='replace', index=False)


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