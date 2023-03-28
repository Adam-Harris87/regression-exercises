import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import env
import matplotlib.pyplot as plt
import seaborn as sns

def acquire_zillow_sfr():
    '''
    This function will retrieve zillow home data for 2017 properties. It will only get
    single family residential properties. the function will attempt to open the data from 
    a local csv file, if one is not found, it will download the data from the codeup
    database. An env file is needed in the local directory in order to run this file.
    '''
    # check to see if there is a csv of the dataset saved in the local directory
    if os.path.exists('zillow_2017_sfr.csv'):
        print('opening data from local file')
        df = pd.read_csv('zillow_2017_sfr.csv', index_col=0)
    # if there is no local data, connect to codeup server and retrive the data
    else:
        # run sql query and write to csv
        print('local file not found')
        print('retrieving data from sql server')
        query = '''
    SELECT 
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid IN(
        SELECT propertylandusetypeid
        FROM propertylandusetype
        WHERE propertylandusedesc = "Single Family Residential")
    -- LIMIT 1000
    ;
        '''
        connection = env.get_db_url('zillow')
        df = pd.read_sql(query, connection)
        # write the data to a local csv file
        df.to_csv('zillow_2017_sfr.csv')

    return df

def clean_zillow_sfr(df):
    '''
    this function will take in a DataFrame of zillow single family resident data,
    it will then remove rows will null values, then remove rows with 0 bedrooms or 
    0 bathrooms, it will then change dtypes of bedroomcnt, calculatedfinishedsquarefeet,
    taxvaluedollarcnt, yearbuilt, and fips to integer, then return the cleaned df
    '''
    # drop rows with null values
    df = df.dropna()
    # change dtypes of columns to int
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
    df.yearbuilt = df.yearbuilt.astype(int)
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype(int)
    df.fips = df.fips.astype(int)
    # drop rows with 0 bedrooms or 0 bathrooms
    mask = (df.bathroomcnt == 0) | (df.bedroomcnt == 0)
    df = df[~mask]
    # there is a house with erronious data 
    # with 4 bedrooms and 4 bathrooms and 952,576 sq feet. lets drop that
    mask = df.calculatedfinishedsquarefeet == df.calculatedfinishedsquarefeet.max()
    df = df[~mask]
    # return the cleaned dataFrame
    return df

def split_zillow(df):
    '''
    this function will take in a cleaned zillow dataFrame and return the data split into
    train, validate and test dataframes in preparation for ml modeling.
    '''
    train_val, test = train_test_split(df,
                                      random_state=1342,
                                      train_size=0.8)
    train, validate = train_test_split(train_val,
                                      random_state=1342,
                                      train_size=0.7)
    return train, validate, test

def wrangle_zillow():
    '''
    This function will acquire the zillow dataset, clean the data, then split the data
    into train, validate and test DataFrames.
    '''
    return split_zillow(
        clean_zillow_sfr(
            acquire_zillow_sfr()))