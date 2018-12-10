import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from numpy import array, dot, transpose
from numpy.linalg import inv
from sklearn import datasets


def linear_regression(x_train, y_train, x_test):
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)

    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)

    predictions = []
    x_test = np.array(x_test)
    for i in x_test:
        components = w[1:] * i
        predictions.append(sum(components) + w[0])
    predictions = np.asarray(predictions)
    return predictions


# def fit(X, y):
#     X = np.insert(X, 0, 1, axis=1)
#
#     U, S, V = np.linalg.svd(X.T.dot(X))
#     S = np.diag(S)
#     X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
#     w = X_sq_reg_inv.dot(X.T).dot(y)


def main():
    dataset = datasets.load_boston()
    target = dataset.target

    f_plot = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    f_plot['Target'] = target
    print(f_plot.info())
    plt.figure(figsize=(15, 15))
    # sns.heatmap(f_plot.corr(), annot=True, cmap="YlGnBu")
    f_plot.plot.scatter(x='LSTAT', y='Target', figsize=(15, 15), grid=True)
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['MDV'] = dataset.target

    df = df.iloc[np.random.permutation(len(df) - 1)].reset_index(drop=1)  # To shuffle the dataset
    train_size = int(round(len(df) * 0.75))  # Training set size: 75% of full data set.
    train = df[:train_size]
    test = df[train_size:]

    x_train = train.drop('MDV', axis=1)
    y_train = train['MDV']

    x_test = test.drop('MDV', axis=1)
    y_test = test['MDV']

    res = linear_regression(x_train, y_train, x_test)
    rss = ((res - y_test)**2).sum()
    print(rss)

#     now we'll use SVD for our data and compute the rss for comparison
    A = f_plot.values

    temp = A.T.dot(A)
    S, V = np.linalg.eig(temp)
    S = np.diag(np.sqrt(S))

    U = A.dot(V).dot(np.linalg.inv(S))
    reconstructed_2 = U.dot(S).dot(V.T)
    df_2 = pd.DataFrame(reconstructed_2, columns=f_plot.columns)
    plt.figure(figsize=(15, 15))
    # sns.heatmap(df_2.corr(), annot=True, cmap="YlGnBu")
    df_2.plot.scatter(x='LSTAT', y='Target', figsize=(15, 15), grid=True)
    plt.show()

    train_2 = df_2[:train_size]
    test_2 = df_2[train_size:]

    x_train_2 = train_2.drop('Target', axis=1)
    y_train_2 = train_2['Target']

    x_test_2 = test_2.drop('Target', axis=1)
    y_test_2 = test_2['Target']

    res_2 = linear_regression(x_train_2, y_train_2, x_test_2)
    rss_2 = ((res_2 - y_test_2)**2).sum()
    print(rss_2)
    plt.savefig(r"figure_1.png")


main()
