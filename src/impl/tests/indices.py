import numpy as np

if __name__ == '__main__':
    mask = np.array([[0,0,1,1],
                    [0,0,1,0],
                    [0,1,1,1]])
    y, x = np.indices(mask.shape, dtype=np.int8)
    print(x)
    print(y)
    # x_positions = x[mask]
    # y_positions = y[mask]
    # print(x_positions)

    y_positions, x_positions = np.nonzero(mask)
    print(x_positions)
    print(y_positions)