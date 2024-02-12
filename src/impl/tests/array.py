import numpy as np
if __name__ == '__main__':
    a = np.array([[5,5,5],
                  [10,10,10],
                  [15,15,15]])
    mask =np.array([[1,0,0],
                    [0,1,1],
                    [0,1,1]])

    print(np.mean(a[mask.nonzero()]))
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    int_mask = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]])

    # Use integer indexing to get the values from arr based on the int_mask
    values = arr[int_mask.nonzero()]

    # Calculate the average value
    average_value = np.mean(values)

    print("Values from arr:", values)
    print("Average value:", average_value)

    print("arr mean: ", np.mean(arr))