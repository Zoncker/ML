import nNn.nn as nn
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import nNn.conjugate_gradient as con_gr
import nNn.classification_map as cmap
np.random.seed(271828182)


def main():

    def gradient_wrapper(weights):
        return NN.cost_grad(weights, x, y)

    def cost_wrapper(weights):
        return NN.cost(weights, x, y)

    def classifier_wrapper(x, y):
        inp = np.atleast_2d([x, y])
        result = NN.forward(inp)
        return np.sign(result)

    x1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=1)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, - y2 + 1))

    max_value = np.max(np.fabs(x))
    x /= (max_value * 1.1)
    y = np.atleast_2d(y).T
    y[y == 0] = -1

    NN = nn.NeuralNetwork([2, 10, 10, 4, 1], ["tanh", "tanh", "tanh", "tanh"])

    w_hat = con_gr.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=600)
    NN.params(w_hat)

    # count errors
    cnt = 0
    N = x.shape[0]
    for i in range(x.shape[0]):
        px, py = x[i, :]
        if y[i, 0] != classifier_wrapper(px, py):
            cnt += 1
    print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

    cmap.plot(classifier_wrapper, x, y.ravel())


main()
