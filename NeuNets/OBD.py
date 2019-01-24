import nNn.nn as nn
import numpy as np
import nNn.conjugate_gradient as conjugate_gradient
import nNn.classification_map as classification_map

from sklearn.datasets import make_gaussian_quantiles

np.set_printoptions(formatter={'float': lambda x: '%.4f' % x})
np.random.seed(271828182)

X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
Y = np.concatenate((y1, - y2 + 1))
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1

NN = nn.NeuralNetwork([2, 10, 1], ["tanh", "tanh"])


def SecondDerivatives(NN, weights, X, Y, h=1e-5):
    grad_zero = NN.cost_grad(weights, X, Y)
    
    grads_forward = np.empty((weights.size, weights.size))
    for i in range(weights.size):
        old = weights[i]
        weights[i] += h
        grads_forward[i,...] = NN.cost_grad(weights, X, Y)
        weights[i] = old
    
    res = np.empty(weights.size)
    for i in range(weights.size):
        gij = grads_forward
        res[i] = (grads_forward[i,i] - grad_zero[i]) / h
    return res


def Salience(NN, weights, X, Y):
    sd = SecondDerivatives(NN, weights, X, Y)
    res = np.zeros(weights.size)
    for i in range(weights.size):
        res[i] = weights[i] ** 2 * sd[i]
    return res


def OBD():
    NN = nn.NeuralNetwork([2, 7, 4, 1], ["tanh", "tanh", "tanh"])

    def gradient_wrapper(weights):
        return NN.cost_grad(weights, X, Y)

    def cost_wrapper(weights):
        return NN.cost(weights, X, Y)

    for i in range(5):
        w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
        s = Salience(NN, w_hat, X, Y)
        useless = np.argmin(np.abs(s))
        w_hat[useless] = 0
        print("Removing axon {} with salience {}".format(useless, s[useless]))
        NN.params(w_hat)
    return NN


NN = OBD()


# count errors
def classifier_wrapper(x, y):
    inp = np.atleast_2d([x,y])
    result = NN.forward(inp)
    return np.sign(result)


cnt = 0
N = X.shape[0]
for i in range(X.shape[0]):
    px, py = X[i, :]
    if Y[i, 0] != classifier_wrapper(px, py):
        cnt += 1
print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

classification_map.plot(classifier_wrapper, X, Y.ravel())

