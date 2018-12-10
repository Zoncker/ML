import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Lasso
# %matplotlib inline
plt.rcParams["figure.figsize"] = (10, 8)


def SolveRidgeRegression(X, y):
    wRR_list = []
    df_list = []
    for i in range(0, 5001, 1):
        lam_par = i
        xtranspose = np.transpose(X)
        xtransx = np.dot(xtranspose, X)
        if xtransx.shape[0] != xtransx.shape[1]:
            raise ValueError('Needs to be a square matrix for inverse')
        lamidentity = np.identity(xtransx.shape[0]) * lam_par
        matinv = np.linalg.inv(lamidentity + xtransx)
        xtransy = np.dot(xtranspose, y)
        wRR = np.dot(matinv, xtransy)
        _, S, _ = np.linalg.svd(X)
        df = np.sum(np.square(S) / (np.square(S) + lam_par))
        wRR_list.append(wRR)
        df_list.append(df)
    return wRR_list, df_list


def makeDFPlots(dfArray, wRRArray, wRR_list):

    fig = plt.figure()
    labels = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    for i in range(0, wRR_list[0].shape[0]):
        plt.plot(dfArray, wRRArray[:,i], )
        plt.scatter(dfArray, wRRArray[:,i], s=8, label=labels[i])
    # df(lambda)
    plt.xlabel(r"df($\lambda$)")
    # and a legend
    plt.legend(loc='lower left')
    plt.show()
    fig.savefig('dfl.png', dpi='figure')


def plotRMSEValue(max_lamda, RMSE_list, poly):
    # colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
    # legend = ["Polynomial Order, p = 1", "Polynomial Order, p = 2", "Polynomial Order, p = 3"]
    plt.plot(range(len(RMSE_list)), RMSE_list, )
    plt.scatter(range(len(RMSE_list)), RMSE_list, s=8)
    # df(lambda)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    # and a legend
    plt.legend(loc='upper left')
    plt.title(r"RMSE vs $\lambda$ values for the test set, $\lambda$ = 0..%d" % max_lamda)


def getRMSEValues(X_test, y_test, wRRArray, max_lamda, poly):
    RMSE_list = []
    for lamda in range(0, max_lamda+1):
        wRRvals = wRRArray[lamda]
        y_pred = np.dot(X_test, wRRvals)
        RMSE = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE_list.append(RMSE)
    plotRMSEValue(max_lamda, RMSE_list, poly=poly)


def main():
    dataset = datasets.load_boston()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['MDV'] = dataset.target

    df = df.iloc[np.random.permutation(len(df) - 1)].reset_index(drop=1)  # To shuffle the dataset
    train_size = int(round(len(df) * 0.75))  # Training set size: 75% of full data set.
    train = df[:train_size]
    test = df[train_size:]

    # X_train = np.genfromtxt('X_train.csv', delimiter=',')
    # y_train = np.genfromtxt('Y_train.csv')

    x_train = train.drop('MDV', axis=1)
    y_train = train['MDV']

    wRR_list, df_list = SolveRidgeRegression(x_train, y_train)
    wRRArray = np.asarray(wRR_list)
    dfArray = np.asarray(df_list)

    makeDFPlots(dfArray, wRRArray, wRR_list)

    # Xn = pd.Series([1] * len(test))
    # test.reset_index(drop=1, inplace=True)
    # x_test = pd.concat([Xn, test.drop('MDV', axis=1)], axis=1)
    x_test = test.drop('MDV', axis=1)
    y_test = test['MDV']

    # X_test = np.genfromtxt('X_test.csv', delimiter=',')
    # y_test = np.genfromtxt('Y_test.csv')

    fig = plt.figure()
    getRMSEValues(x_test, y_test, wRRArray, max_lamda=30, poly=1)
    plt.show()
    fig.savefig('res.png', dpi='figure')


main()
