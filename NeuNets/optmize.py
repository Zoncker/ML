import nNn.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import nNn.conjugate_gradient as conjugate_gradient
from sklearn.datasets import make_gaussian_quantiles
import nNn.classification_map as cmap


np.random.seed(27182818)
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
Y = np.concatenate((y1, - y2 + 1))

max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1


def main():
    def AddNeuron(NN):
        a, b, c = NN.layers
        NN.layers[1] += 1
        NN.b[0] = np.concatenate((NN.b[0], [0]))
        n = NN.W[0].shape[1]
        NN.W[0] = np.hstack((NN.W[0], np.random.randn(a, 1) * 1e-5))
        NN.W[1] = np.vstack((NN.W[1], np.random.randn(1, c) * 1e-5))

    def classifier_wrapper(x, y):
        inp = np.atleast_2d([x, y])
        result = NN.forward(inp)
        return np.sign(result)

    start = 3
    NN = nn.NeuralNetwork([2, start, start, 1], ["tanh", "tanh", "tanh"])
    # AddNeuron(NN)
    # quit()

    def gradient_wrapper(weights):
        return NN.cost_grad(weights, X, Y)

    def cost_wrapper(weights):
        return NN.cost(weights, X, Y)


    def trained_cost():
        w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
        return NN.cost(w_hat, X, Y)

    now = start
    neurons = []
    costs = []
    for i in range(8):
        print("Making {} neurons".format(now))
        neurons.append(now)
        costs.append(trained_cost())
        now += 1
        AddNeuron(NN)
    cmap.plot(classifier_wrapper, X, Y.ravel())
    fig1 = plt.figure()
    plt.plot(neurons, costs)
    plt.xlabel("Number of neurons")
    plt.ylabel("Error value")
    fig1.savefig('errorvalue.png', dpi='figure')
    plt.show()


    # count errors
    # cnt = 0
    # N = X.shape[0]
    # for i in range(X.shape[0]):
        # px, py = X[i,:]
        # if (Y[i,0] != classifier_wrapper(px, py)):
            # cnt += 1
    # print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

main()

