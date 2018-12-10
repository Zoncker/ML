import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from utils import gauss, quartic,epanenchenkov, r1_dist, train_test_split
from regress import Regression


def main(start_prec, step):

    diabetes = datasets.load_diabetes()
    x_tr, y_tr, x_te, y_te, to_test = train_test_split(diabetes, featr=3, n=20)

    r_quart = Regression(quartic, r1_dist, 0)
    r_gauss = Regression(epanenchenkov, r1_dist, 0)
    rlowess_quart = Regression(quartic, r1_dist, 1)

    r_quart.fit1(x_tr, y_tr, start_prec, step)
    r_gauss.fit1(x_tr, y_tr, start_prec, step)
    rlowess_quart.fit(x_tr, y_tr, start_prec, step)

    res1 = r_quart.predict(to_test)
    res2 = r_gauss.predict(to_test)
    res3 = rlowess_quart.predict(to_test)

    print("SSE NW quartic: %.2f"
          % np.mean((res1 - y_te) ** 2))
    print("SSE NW epanenchnekov: %.2f"
          % np.mean((res2 - y_te) ** 2))
    print("SSE LOWESS quart: %.2f"
          % np.mean((res3 - y_te) ** 2))
    fig = plt.figure()
    plt.plot(x_tr, y_tr, 'r.', markersize=5, color='brown')
    plt.plot(x_te, y_te, 'r.', markersize=7, color='pink')
    plt.plot(to_test, res1, color='magenta', linewidth=2)
    plt.plot(to_test, res2, color='cyan', linewidth=2)
    plt.plot(to_test, res3, color='blue', linewidth=2)

    plt.legend(("Train data", "Test data", "NW quartic", "NW epanenchnekov",
                "LOWESS quart"), loc="upper left")
    # "Lowess quartic kernel"
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xticks(())
    plt.yticks(())

    plt.show()
    fig.savefig('res.png', dpi='figure')


main(0.05, 0.01)
