import math


def epanenchenkov(r):
    if abs(r) > 1.:
        return 0.
    return 0.75 * (1 - r ** 2)


def gauss(r):
    return ((2. * math.pi) ** (-0.5)) * math.exp(-0.5 * (r ** 2))


def rectangle(r):
    if abs(r) > 1.:
        return 0.
    return 0.5


def triangle(r):
    if abs(r) > 1:
        return 0.
    return 1. - abs(r)


def quartic(r):
    if abs(r) > 1:
        return 0.
    return (1. - r ** 2) ** 2