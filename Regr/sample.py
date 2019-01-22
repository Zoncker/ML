from sklearn import model_selection
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


class Lasso_t(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = deepcopy(beta)
                tmp_beta[j] = 0.0
                r_j = y - np.dot(X, tmp_beta)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self._soft_thresholding_operator(arg1, arg2) / (X[:, j] ** 2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    def predict(self, X):
        y = np.dot(X, self.coef_)
        if self.fit_intercept:
            y += self.intercept_ * np.ones(len(y))
        return y


alphas = 4**np.linspace(10, -7, 1000)
lasso = Lasso(max_iter=1000, normalize=True)
coefs = []

dataset = datasets.load_boston()
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std()
X = df.iloc[:, :13].values
res = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X), y)
    coefs.append(lasso.coef_)
    res.append(lasso.score(scale(X), y))
    # print(res)


fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.plot(alphas * 2, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.legend(("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"),
           loc='upper right')
plt.show()
fig.savefig('lasso_s.png', dpi='figure')
fig1 = plt.figure(figsize=(10, 8))
plt.plot(range(len(res)), res, linewidth=3, color='green')
plt.xlabel(r"$\alpha$")
plt.ylabel("$R^2$ score")
plt.title(r"$R^2$ vs $\alpha$ values for the test set")
plt.show()
fig1.savefig('lasso_r2.png', dpi='figure')

