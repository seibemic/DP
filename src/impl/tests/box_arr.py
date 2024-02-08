import numpy as np
if __name__ == '__main__':
    A = np.ones((5, 5))
    x, y, xx, yy = 1, 1, 3,4
    m1=np.mean(A[x:xx+1,y:yy+1])
    print(m1)
    A[x:xx+1,y:yy+1] = 0
    m1 = np.mean(A[x:xx + 1, y:yy + 1])
    print("A")
    print(A)
    B = np.ones((5,5))
    print("A * B")
    print(A * B)

    a = np.ma.array([1, 2, 3], mask=  np.invert([False, False, True]))
    print(a)
    print(a.mean())
    dt = 1 / 25

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    print(F @ np.ones(4))