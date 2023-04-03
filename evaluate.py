import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(y, yhat):
    plt.scatter(y, (yhat - y))

    # The lineplot is my regression line used for predictions
    plt.axhline(0, color='red')

    plt.xlabel('actual tax value')
    plt.ylabel('predicted tax value')
    plt.title('residual plot')
    plt.show()

def regression_errors(y, yhat): 
    '''takes in array actual and predicted values.
    returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)'''
    
#     train['residual'] = train.tax_value - train.yhat
#     train['residual_2'] = (train.residual) ** 2
#     SSE = train.residual_2.sum()
    SSE = sum((y - yhat) ** 2)
    
    # Total Sum of Squares = SSE for baseline
#     TSS = SSE_baseline
    TSS = sum((y - y.mean()) ** 2)
    
    # ESS - Explained Sum of Squares ('Explained Error')
    ESS = TSS - SSE
    
    MSE = SSE / len(y)
    
    RMSE = MSE ** (1/2)
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y): 
    '''computes the SSE, MSE, and RMSE for the baseline model'''
    SSE = sum((y - y.mean()) ** 2)
    MSE = SSE / len(y)
    RMSE = MSE ** (1/2)
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat): 
    '''returns true if your model performs better than the baseline, otherwise false'''
    SSE = sum((y - yhat) ** 2)
    MSE = SSE / len(y)
    RMSE = MSE ** (1/2)
    
    SSE_base = sum((y - y.mean()) ** 2)
    MSE_base = SSE_base / len(y)
    RMSE_base = MSE_base ** (1/2)
    
    return RMSE < RMSE_base