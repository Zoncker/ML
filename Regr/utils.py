import math
import numpy as np


def gauss(r):
    return ((2. * math.pi) ** (-0.5)) * math.exp(-0.5 * (r ** 2))


def quartic(r):
    if abs(r) > 1:
        return 0.
    return (1. - r ** 2) ** 2


def r1_dist(a, b):
    return abs(float(a - b))


def epanenchenkov(r):
    if abs(r) > 1.:
        return 0.
    return 0.75 * (1 - r ** 2)


def rectangle(r):
    if abs(r) > 1.:
        return 0.
    return 0.5


def train_test_split(dataset, featr, n):

        diabetes_x = dataset.data[:, np.newaxis, featr]

        diabetes_x_train = diabetes_x[:-n]
        diabetes_x_test = diabetes_x[-n:]

        diabetes_y_train = dataset.target[:-n]
        diabetes_y_test = dataset.target[-n:]

        to_test = np.sort(diabetes_x_test, axis=None)

        return diabetes_x_train, diabetes_y_train, diabetes_x_test, diabetes_y_test, to_test


