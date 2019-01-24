import numpy as np
import matplotlib.pyplot as plt


def plot(classifier, inp, out, ticks=200):
    # ranges
    xfrom = inp[:, 0].min() * 1.1
    xto = inp[:, 0].max() * 1.1
    yfrom = inp[:, 1].min() * 1.1
    yto = inp[:, 1].max() * 1.1

    # meshgrid
    h = (xto - xfrom) / ticks
    xx, yy = np.arange(xfrom, xto, h), np.arange(yfrom, yto, h)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.empty(xx.shape, dtype=float)

    # classify mgr
    pos = 0
    for x in range(xx.shape[0]):
        for y in range(xx.shape[1]):
            zz[x][y] = classifier(xx[x][y], yy[x][y])
    fig1 = plt.figure()
    plt.clf()
    # ('brown', 'pink')
    plt.contourf(xx, yy, zz, alpha=0.5, colors=('red', 'blue', 'cyan'))  # division line
    plt.scatter(inp[:, 0], inp[:, 1], c=out, s=50)  # dataset points
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.legend(("division line", "class1", "class2"), loc="upper left")
    plt.show()
    fig1.savefig('clmap2.png', dpi='figure')
