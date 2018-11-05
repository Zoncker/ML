from sklearn import datasets
import numpy as np

def main():

    boston = datasets.load_boston()
    print(boston.data.shape)
    # arr = np.arange(15)
    # esc = arr[:, np.newaxis, 2]
    b_x = boston.data[:, np.newaxis, 3]

    print(boston.data)
    print(b_x)
    print(b_x.shape)
    # print(esc)

main()
