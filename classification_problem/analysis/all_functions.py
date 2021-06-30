import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split


def merge_data(cust_data, prod_data, assort_data):
    """ Merge customer features, product features and assortment file
    and return thr merged data frame
    """
    # convert to long format based on different genre for each customer
    rows = []
    for i, row in cust_data.iterrows():
        values = [each.strip("[]\"\' ") for each in row.favorite_genres.split(',')]
        for a in values:
            if not a:
                a = None
            rows.append([row.customer_id, row.age_bucket, row.is_returning_customer, a])

    # customer data
    customer = pd.DataFrame(rows, columns=cust_data.columns)
    customer.columns = ['customer_id', 'age_bucket', 'is_returning_customer', 'genre']

    # make sure we have all the rows from assortment file
    prod_assort = assort_data.merge(prod_data, on='product_id', how='left')
    prod_assort = prod_assort.fillna(-2)   # fill null values with -2 (create a new category)

    # merge data with customer_purchased file
    merged_df = customer.merge(prod_assort, on=['customer_id', 'genre'], how='right')
    merged_df = merged_df.fillna(-2)

    return merged_df


def model_validation(classifier, X, y, n_folds=5):
    """Validate the model by cross validation"""
    all_tpr = []
    all_fpr = []
    all_roc_auc = []
    skf = StratifiedKFold(n_splits=n_folds)
    for i, (train, test) in enumerate(skf.split(X, y)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        roc_auc = auc(fpr, tpr)
        print("AUC for fold ", (i, roc_auc))
        all_roc_auc.append(roc_auc)

    # plot ROC curve for all folds
    plot_ROC_curve(all_fpr, all_tpr, all_roc_auc, n_folds)


def plot_ROC_curve(all_fpr, all_tpr, all_roc_auc, n_folds):
    """Plot the ROC curve"""

    for fold in range(n_folds):
        plt.plot(all_fpr[fold], all_tpr[fold], lw=1, label='ROC fold %d (area = %0.2f)' % (fold, all_roc_auc[fold]))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for cv folds')
    plt.legend(loc="lower right")
    plt.savefig('plots/ROC_cv.png')
    plt.close()


def model_metrics(classifier, X, y):
    """Get the metrics like fnr, fpr, accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = classifier.fit(X_train, y_train)
    probas_ = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)

    # plot fpr for different thresholds
    plt.plot(thresholds, fpr)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Thresholds')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate for different thresholds')
    plt.savefig('plots/fpr.png')
    plt.close()

    # plot fnr for different thresholds
    fnr = 1 - tpr
    plt.plot(thresholds, fnr)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Thresholds')
    plt.ylabel('False Negative Rate')
    plt.title('False Negative Rate for different thresholds')
    plt.savefig('plots/fnr.png')
    plt.close()

    # plot accuracy for different thresholds
    all_acc_scores = []
    thres_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thres_values:
        y_pred = probas_[:, 1] > threshold
        print("Confusion matrix for threshold\n", threshold)
        print(confusion_matrix(y_test, y_pred))
        all_acc_scores.append(accuracy_score(y_test, y_pred))
    plt.plot(thres_values, all_acc_scores)
    plt.xlabel('Thresholds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different thresholds')
    plt.savefig('plots/accuracy.png')
    plt.close()


def check_loan_repay_status(original_order, next_order, last_assort, next_assort):
    """Function that outputs boolean value whether company will be pay back loan
    as well as afford next purchase order"""

    print("######### TOTAL AMOUNT SPEND ######################")
    print("\n\nGet total loan amount")
    # Get a new column - amount spend for each (customer, product) pair
    original_order['amount_spend'] = original_order['quantity_purchased'] * original_order['cost_to_buy']
    loan_amount = original_order['amount_spend'].sum()
    print("Total loan amount is", loan_amount)

    print("\nGet total amount spend on next purchase order")
    next_order['amount_spend'] = next_order['quantity_purchased'] * next_order['cost_to_buy']
    total_amount_next_purchase = next_order['amount_spend'].sum()
    print(total_amount_next_purchase)

    print("\nTotal amount spend - (Initial Loan + Next purchase order)")
    total_amount_spend = loan_amount + total_amount_next_purchase
    print(total_amount_spend)

    # concatenate both files
    total_spend = pd.concat([original_order, next_order])
    total_spend_by_product = total_spend.groupby('product_id', as_index=False). \
        agg({"cost_to_buy": "mean", "retail_value": "mean"})

    print("############## TOTAL AMOUNT RECEIVED ###################")
    # get purchase info from last and next assortment
    assort = pd.concat([last_assort, next_assort])
    complete_assort = assort.merge(total_spend_by_product, on='product_id', how='left')
    ship_cost = 0.60
    # get a new column ''
    complete_assort['amount_received'] = np.where(complete_assort['purchased'] == 1,
                                                  complete_assort['retail_value'] - ship_cost,
                                                  -2 * ship_cost)

    total_amount_received = complete_assort['amount_received'].sum()
    print("total amount received is ", total_amount_received)
    profit = total_amount_received - total_amount_spend
    print("\nProfit of the company is", profit)

    return profit
