import numpy as np
from utils import rectangle


class Regression:
    def __init__(self, kernel_f, dist_f, ident):
        self._kernel = kernel_f
        self._dist = dist_f
        self._h = 1
        self._coefs = []
        self._gammas = []
        self._ident = ident

    def predict(self, x):
        res = []
        if self._ident is 1:
            for x in x:
                res.append(self.__kern_smooth(x, self._x, self._y, self._kernel, self._dist, self._h, self._gammas))
            return res
        else:
            for x in x:
                res.append(self.__kern_smooth(x, self._x, self._y, self._kernel, self._dist, self._h))
            self._coefs = res
            return self._coefs

    def fit1(self, x, y, precision=0.007, step=0.04):
        h = precision
        # if self._ident is 0:
        cur_loo = self.__loo(x, y, h)
        min_loo = cur_loo
        min_h = h
        while cur_loo > precision and h > step:
            h -= step
            print(h)
            cur_loo = self.__loo(x, y, h)
            if min_loo > cur_loo:
                min_loo = cur_loo
                min_h = h
        self._h = min_h
        self._x = x
        self._y = y
        print(min_h)

    def fit(self, x, y, precision=0.007, step=0.04):
        h = precision
        prev_gammas = [0] * len(y)
        cur_gammas = [1] * len(y)
        cur_loo = self.__loo(x, y, h, cur_gammas)
        min_loo = cur_loo
        min_h = h
        while cur_loo > precision and h > step:
            h -= step
            print(h)

            idx = 0
            while prev_gammas != cur_gammas:
                prev_gammas = cur_gammas
                for xi, yi in zip(x, y):
                    newX = np.delete(x, xi)
                    newY = np.delete(y, yi)
                    parameter = self.__kern_smooth(xi, newX, newY, self._kernel, self._dist, h, cur_gammas)
                    value = rectangle(abs(parameter - yi))
                    cur_gammas[idx] = value
                    idx += 1
            cur_loo = self.__loo(x, y, h, cur_gammas)
            if min_loo > cur_loo:
                min_loo = cur_loo
                min_h = h

        print("LOWESS MIN H", min_h)
        self._gammas = cur_gammas
        self._h = min_h
        self._x = x
        self._y = y

    def __kern_smooth(self, cur_x, x, y, kernel_func, dist_func, h, g=None):
        numerator = 0
        denominator = 0
        res = 0
        if g is not None:
            for xi, yi, gi in zip(x, y, g):
                dist = dist_func(cur_x, xi)
                core_value = kernel_func(dist / h) * gi
                denominator += core_value
                numerator += yi * core_value
        else:
            for xi, yi in zip(x, y):
                dist = dist_func(cur_x, xi)
                core_value = kernel_func(dist / h)
                denominator += core_value
                numerator += yi * core_value
        if denominator > 0:
            res = numerator / denominator
        return res

    def __loo(self, x, y, h, g=None):
        res = 0
        count = 1
        for xi, yi in zip(x, y):
            new_x = np.delete(x, xi)
            new_y = np.delete(y, yi)

            if g is not None:
                parameter = self.__kern_smooth(xi, new_x, new_y, self._kernel, self._dist, h, g)
            else:
                parameter = self.__kern_smooth(xi, new_x, new_y, self._kernel, self._dist, h)

            value = (parameter - yi) ** 2
            res += value
            count += 1

        return res
