from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def rank_scaled_dict(rank, names, order=1):
    """ Scale the rankings from different models and add
    them all to a dictionary for comparison
    """
    # scale the scores between 0 and 1
    scaled_rank = MinMaxScaler().fit_transform(order*np.array([rank]).T).T[0]
    final_ranks = map(lambda x: round(x, 3), scaled_rank)
    adict = dict(zip(names, final_ranks))
    print([(k, v) for k, v in sorted(adict.items(), key=lambda x: x[1], reverse=True) if abs(v) > 0.0])

    return dict(zip(names, final_ranks))


def rank_features(X, y, collinear):
    """ Returns most important features based on different techniques
    """
    rank = {}

    col_names = X.columns

    print("\n\nStandarized variables before using different models for ranking")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y).ravel()

    print("\n\nRanking from univariate regression based on F-scores")  # only identifies linear relationship
    f, pval = f_regression(X_scaled, y_scaled)
    rank["Corr"] = rank_scaled_dict(f, col_names)
    print("\nAll very significant features with p-val < 0.01 from univariate regression")
    pvals_dict = dict(zip(col_names, pval))
    print([(k, v) for k, v in sorted(pvals_dict.items(), key=lambda x: x[1]) if v < 0.01])

    if not collinear:
        lr = LinearRegression().fit(X_scaled, y_scaled)
        print("\n\nRanking based on linear regression")
        rank["Linear Regression"] = rank_scaled_dict(np.abs(lr.coef_), col_names)

    print("\n\nRanking based on lasso ( L1 penalty) based on very low regularization value alpha - 0.001")
    lasso = Lasso(alpha=0.001).fit(X_scaled, y_scaled)
    rank['Lasso'] = rank_scaled_dict(lasso.coef_, col_names)

    print("\n\nRanking based on Random forests variable importance graph; using mutual information")
    rf = RandomForestRegressor().fit(X_scaled, y_scaled)
    rank["Random Forest"] = rank_scaled_dict(rf.feature_importances_, col_names)

    # Get the mean score based on all the algorithms
    temp = {}
    for col in col_names:
        temp[col] = round(np.mean([rank[algo][col] for algo in rank.keys()]), 3)

    rank['avg_scores'] = temp
    # print(rank)
    methods = sorted(rank.keys())

    print("\n\nRanking based on different metrics")
    sorted_colnames = sorted(rank['avg_scores'], key=rank['avg_scores'].__getitem__, reverse=True)
    print("\t%s" % "\t".join(methods))
    for name in sorted_colnames:
        print("%s\t%s" % (name, "\t\t".join(map(str, [rank[method][name] for method in methods]))))

    select_features = [k for k, v in rank['avg_scores'].items() if v > 0]
    print("\n\nImportant predictors are based on average score:", select_features)

    print("\n\n One can also use Recursive feature elimination for feature engineering,"
          "but given so many features, this would take long time.")

    return select_features

