"""
Feature selection process

This program takes data as input and returns important features
that can help predict the response variable

- Things being considered here:
Missing values
Variance Threshold
Effects of multi-collinearity
Finally, ranking of features based on various techniques like
    - Correlation
    - Lasso (Penalized regression)
    - Random Forests
    - Linear Regression (only if multi-collinearity is not present in the data)
"""

import argparse
import sys
import pandas as pd
from .all_functions import *

np.set_printoptions(suppress=True)


def main():

    # redirects all prints to log file
    temp = sys.stdout
    sys.stdout = open('log.txt', 'w')

    # Please specify these parameters to proceed: not mandatory
    parser = argparse.ArgumentParser(description="Please specify these parameters to proceed:")
    parser.add_argument('--file_name', nargs='?', default='data/linear_regression_challenge.csv')
    parser.add_argument('--num_features', nargs='?', const=1000, default=1000)
    args = parser.parse_args()
    filename = args.file_name
    num_features = int(args.num_features)

    y_feature = ['y']
    x_features = ['x' + str(i) for i in range(num_features)]

    # read data
    data = pd.read_table(filename, sep='|', names=y_feature + x_features)
    print("Shape of data is", data.shape)

    # Cleaning data
    print("\n\nCheck for Missing values")
    percent_rows_nan = (data.shape[0] - data.dropna().shape[0])/data.shape[0]
    print("Percentage of rows with missing values", percent_rows_nan)

    # If percent of missing values in data is greater than 10%, impute missing values
    # else drop those rows
    if 0.0 < percent_rows_nan <= 0.2:
        # Dropped rows with missing values
        data = data.dropna()
        print("Shape of data after removing missing values is", data.shape)
    elif percent_rows_nan > 0.2:
        # Dropped rows where all values is that row is missing
        # and Imputed rows with fewer missing values
        data = data.dropna(how='all')
        data = data.apply(lambda x: x.fillna(x.mean()))
        print("Shape of data after imputing missing values is", data.shape)

    # feature selection
    print("\n\n Remove features with low variance")
    threshold = 0.2

    columns_to_remove = data.std()[data.std() < threshold].index.values
    x_features = [col for col in x_features if col not in columns_to_remove]
    print("Shape of data now is ", data.shape)

    print("\n\n Check for multicollinearity by checking pairwise correlation between features. "
          "If multicollinearity is high, then we can remove the ranking based on linear regression")

    pairwise_cors = data[x_features].corr().abs().stack()

    # remove diagonal and upper triangular pairs
    drop_pairs = set()
    cols = data[x_features].columns
    for i in range(0, data[x_features].shape[1]):
        for j in range(0, i+1):
            drop_pairs.add((cols[i], cols[j]))

    print("Top ten correlations", pairwise_cors.drop(drop_pairs).sort_values(ascending=False)[0:10])
    highest_pairwise_cor = pairwise_cors.drop(drop_pairs).sort_values(ascending=False)[0]

    if highest_pairwise_cor >= 0.8:
        collinearity_factor = True
    else:
        collinearity_factor = False

    # function that ranks features based on different metrics
    selected_features = rank_features(X=data[x_features], y=data[y_feature], collinear=collinearity_factor)
    sys.stdout.close()
    sys.stdout = temp
    print(" Selected features are", selected_features)


