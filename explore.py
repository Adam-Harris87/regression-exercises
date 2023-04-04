import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, \
f_regression
from scipy.stats import pearsonr, spearmanr

def plot_variable_pairs(df):
    '''
    This function will take in a dataFrame, and will display a pairplot of the variable
    relationships along with a regression line for each pair
    '''
    # take a sample of the dataFrame in order to cut down computing time
    sample = df.sample(1000)
    # create a pairplot
    sns.pairplot(sample, corner=True, kind='reg', plot_kws={'color': 'blue'})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat_cols, cont_cols):
    '''
    This function will take in a DataFrame and a list of categorical and continuous
    variable columns, then display visualizations for each pair of columns
    '''
    # cycle through all the categorical columns
    for cat_col in cat_cols:
        # cycle through all the continuous columns
        for cont_col in cont_cols:
            plt.subplot(311)
            # create a boxplot
            sns.boxplot(x=cat_col, y=cont_col, data=df)
            plt.subplot(312)
            # create a violinplot
            sns.violinplot(x=cat_col, y=cont_col, data=df)
            plt.subplot(313)
            # create a barplot
            sns.barplot(x=cat_col, y=cont_col, data=df)
            plt.show()
            
def select_kbest(X, y, k):
    '''
    This function will return a list of (k) number of columns from the predictors (X)
    for the target variable (y) using the SelectKBest function
    '''
    # make the thing
    kbest = SelectKBest(f_regression, k=k)
    # fit the thing
    _ = kbest.fit(X, y)
    # return a list of the columns chosen as features
    return X.iloc[:,kbest.get_support()].columns.to_list()

def rfe(X, y, k):
    '''
    This function will return a list of (k) number of columns from the predictors (X)
    for the target variable (y) using the RFE function
    '''
    # make a model for the RFE to work on
    model = LinearRegression()
    # make the RFE thing
    rfe = RFE(model ,n_features_to_select=k)
    # fit the RFE to our dataset
    _ = rfe.fit(X, y)
    # return the column list of chosen features
    return X.iloc[:,rfe.get_support()].columns.to_list()