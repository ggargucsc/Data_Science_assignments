import argparse
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .all_functions import *

"""
This program takes data as input and 
returns an answer to the question - 
Will Books&Co be able to both pay back loan and 
afford the next book purchase order?
"""


def main():

    # redirects all prints to log file
    temp = sys.stdout
    sys.stdout = open('log.txt', 'w')

    # Please specify these parameters to proceed: not mandatory
    parser = argparse.ArgumentParser(description="Please specify these parameters to proceed:")
    parser.add_argument('--orig_purchase_file', nargs='?', default='original_purchase_order.csv')
    parser.add_argument('--next_purchase_file', nargs='?', default='next_purchase_order.csv')
    parser.add_argument('--customer_file', nargs='?', default='customer_features.csv')
    parser.add_argument('--product_file', nargs='?', default='product_features.csv')
    parser.add_argument('--last_assortment_file', nargs='?', default='last_month_assortment.csv')
    parser.add_argument('--next_assortment_file', nargs='?', default='next_month_assortment.csv')

    args = parser.parse_args()
    orig_purchase_file = args.orig_purchase_file
    next_purchase_file = args.next_purchase_file
    customer_file = args.customer_file
    product_file = args.product_file
    last_assortment_file = args.last_assortment_file
    next_assortment_file = args.next_assortment_file

    print("Read all the files")
    orig_purchase = pd.read_csv(orig_purchase_file)
    next_purchase = pd.read_csv(next_purchase_file)
    customer_features = pd.read_csv(customer_file)
    product_features = pd.read_csv(product_file)
    last_assortment = pd.read_csv(last_assortment_file)
    next_assortment = pd.read_csv(next_assortment_file)

    print("############## CLEAN DATA ################################# ")
    print("Get merged data from three files\n - customer_features, product_features and last_month_assortment")
    merged_df = merge_data(cust_data=customer_features,
                           prod_data=product_features,
                           assort_data=last_assortment)

    # change data types of columns
    cols_not_cat = ['customer_id', 'product_id', 'length', 'purchased']
    for col in merged_df.columns:
        if col not in cols_not_cat:
            merged_df[col] = merged_df[col].astype('category')

    # One hot encoding for categorical variables
    merged_df = pd.get_dummies(merged_df,
                               columns=['age_bucket', 'is_returning_customer', 'genre', 'difficulty', 'fiction'],
                               prefix=['age', 'return', 'genre', 'diff', 'fic'])

    merged_df.purchased = merged_df.purchased.astype('int')

    print("################## BUILD MODEL using last assortment data ############################")
    # Get data ready for modeling
    X = np.array(merged_df.drop(['customer_id', 'product_id', 'purchased'], axis=1))
    y = merged_df.purchased

    print("Validate the model using 5-fold cross validation")
    model_validation(RandomForestClassifier(), X, y)

    # get the metrics like accuracy, fpr, fnr
    # these parameters slightly improves the performance compared to default model
    model_metrics(RandomForestClassifier(n_estimators=100, max_features=20), X, y)

    # Get the model to be used later for prediction
    rf_model = RandomForestClassifier(n_estimators=100, max_features=20).fit(X, y)

    # choose threshold based on accuracy, fpr and fnr
    threshold = 0.48

    print("################## PREDICTION on next assortment ###########################")
    print("Get merged data from three files - customer_features, product_features and next_month_assortment ")
    data_next_assortment = merge_data(cust_data=customer_features,
                                      prod_data=product_features,
                                      assort_data=next_assortment)

    # change data types of columns
    cols_not_cat = ['customer_id', 'product_id', 'length']
    for col in data_next_assortment.columns:
        if col not in cols_not_cat:
            data_next_assortment[col] = data_next_assortment[col].astype('category')

    # One hot encoding for categorical variables
    data_next_assortment = pd.get_dummies(data_next_assortment,
                                          columns=['age_bucket', 'is_returning_customer', 'genre', 'difficulty', 'fiction'],
                                          prefix=['age', 'return', 'genre', 'diff', 'fic'])

    # Predict probabilities whether customer will purchase or not on next assortment
    probas_ = rf_model.predict_proba(data_next_assortment.drop(['customer_id', 'product_id'], axis=1))
    data_next_assortment['purchased'] = probas_[:, 1] > threshold
    print("Predict whether customer will purchase or not on next assortment\n",
          data_next_assortment['purchased'].value_counts())
    data_next_assortment.purchased = data_next_assortment.purchased.astype('int')

    # Get predicted value for each (customer, product) pair on next assortment file
    next_assortment = pd.read_csv('next_month_assortment.csv')
    next_assort_predicted = data_next_assortment[['product_id', 'customer_id', 'purchased']].\
        merge(next_assortment, on=['product_id', 'customer_id'], how='right')

    print("Check whether the company would be able to repay loan and as well as next purchase order")
    profit = check_loan_repay_status(original_order=orig_purchase,
                                     next_order=next_purchase,
                                     last_assort=last_assortment,
                                     next_assort=next_assort_predicted)

    sys.stdout.close()
    sys.stdout = temp

    if profit >= 0:
        print("Yes")  # able to pay loan and afford next purchase order
    else:
        print("No")
