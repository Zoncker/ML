import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from ker import gauss, quartic
from nadwats import NadarayaWatsonRegression
import math


def distanceR1(a, b):
    return abs(float(a - b))


def distanceR2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def main(start_precision, step):
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 3]
    n = 20
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-n]
    diabetes_X_test = diabetes_X[-n:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-n]
    diabetes_y_test = diabetes.target[-n:]

    to_test = np.sort(diabetes_X_test, axis=None)
    print(to_test)

    regr1 = NadarayaWatsonRegression(quartic, distanceR1)
    regr2 = NadarayaWatsonRegression(gauss, distanceR1)

    regr1.fit(diabetes_X_train, diabetes_y_train, start_precision, step)
    regr2.fit(diabetes_X_train, diabetes_y_train, start_precision, step)

    res1 = regr1.predict(to_test)
    res2 = regr2.predict(to_test)
    '''
    # The mean square error
    print("Sum of squares quartic: %.2f"
          % np.mean((res1 - diabetes_y_test) ** 2))
    print("Sum of squares gauss: %.2f"
          % np.mean((res2 - diabetes_y_test) ** 2))
    '''

    # Plot outputs
    # plt.scatter(diabetes_X_train, diabetes_y_train, color='yellow')
    plt.scatter(to_test, res1,  color='green')
    plt.scatter(to_test, res2,  color='red')
    plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
    # plt.plot(to_test, res1, color='green', linewidth=2)
    # plt.plot(to_test, res2, color='red', linewidth=2)

    plt.xticks(())
    plt.yticks(())

    plt.show()


main(0.05, 0.01)