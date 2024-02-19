import numpy as np


def transition_matrix(pd, pk):
    return np.array([[pd, 1 - pd, 0],
                     [pd, 1 - pd - (1 - pk), 1 - pk],
                     [pd, 1 - pd - (1 - pk), 1 - pk]])
if __name__ == "__main__":
    init = np.array([0,0.1,0.9])
    res = np.identity(3)
    res = res @ transition_matrix(0.9, 0)
    print (res)
    print(init @ res)