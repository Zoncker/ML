import numpy as np


class NadarayaWatsonRegression:
    def __init__(self, kernel_func, dist_func):
        self.__kernel = kernel_func
        self.__dist = dist_func
        self.__h = 1
        self._coef = []

    def predict(self, X):
        res = []
        for x in X:
            res.append(self.__get_parameter(x, self.__X, self.__Y, self.__kernel, self.__dist, self.__h))
        self._coef = res
        return self._coef

    def fit(self, X, Y, precision=0.09, step=0.025):
        h = precision
        cur_loo = self.__loo(X, Y, h)
        min_loo = cur_loo
        min_h = h
        while cur_loo > precision and h > step:
            h -= step
            print(h)
            cur_loo = self.__loo(X, Y, h)
            if min_loo > cur_loo:
                min_loo = cur_loo
                min_h = h
        self.__h = min_h
        self.__X = X
        self.__Y = Y
        print(min_h)

    def __get_parameter(self, cur_x, X, Y, kernel_func, dist_func, h):
        numerator = 0
        denominator = 0
        res = 0
        #self.__sort_by_distance(cur_x, X, Y)
        for xi, yi in zip(X, Y):
            dist = dist_func(cur_x, xi)
            core_value = kernel_func(dist / h)
            denominator += core_value
            numerator += yi * core_value
        if denominator > 0:
            res = numerator / denominator

        return res

    def __loo(self, X, Y, h):
        res = 0
        count = 1
        for xi, yi in zip(X, Y):
            newX = np.delete(X, xi)
            newY = np.delete(Y, yi)
            parameter = self.__get_parameter(xi, newX, newY, self.__kernel, self.__dist, h)
            value = (parameter - yi) ** 2
            res += value
            count += 1

        return res

    def __sort_by_distance(self, cur_x, X, Y):
        list = []
        for xi, yi in zip(X, Y):
            dist = self.__dist(cur_x, xi)
            list.append(dist)
        newX = np.c_[X, Y, list]
        newX = newX[newX[:, 2].argsort()]
        X = newX[:, 0]
        Y = newX[:, 1]