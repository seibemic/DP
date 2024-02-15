import numpy as np

# Define the time-dependent function a(t)
T= np.random.rand(1000)
T= np.array([0.9,0.9])
def pd(t):
    # Your implementation of a(t) goes here
    return T[t]  # Replace this with your actual function

# Define the stochastic matrix P(t) for a given time t
def stochastic_matrix(t):
    return np.array([[1 - pd(t), pd(t)],
                     [pd(t), 1 - pd(t)]])

if __name__ == "__main__":
    np.random.seed(42)

    # Define the number of stages (k)
    k = 100

    # Generate an array with 40 random values between 0.9 and 1
    high_values_1 = np.random.uniform(0.9, 1, 40)

    # Generate an array with 5 random values between 0.2 and 0.3
    low_values_1 = np.random.uniform(0.2, 0.3, 5)

    # Generate an array with 55 random values between 0.9 and 1
    high_values_2 = np.random.uniform(0.9, 1, 5)

    low_values_2 = np.random.uniform(0.2, 0.3, 5)

    high_values_3 = np.random.uniform(0.9, 1, 45)



    # Concatenate the arrays to create random_a_values
    Pd = np.concatenate([high_values_1, low_values_1, high_values_2,low_values_2, high_values_3])

    pk_1 = np.random.uniform(0.8,1, 40)
    pk_2 = np.random.uniform(0.8,1, 5)
    pk_3 = np.random.uniform(0.8, 1, 5)
    pk_4 = np.random.uniform(0, 0.1, 5)
    pk_5 = np.random.uniform(0.8, 1, 45)
    Pk = np.concatenate([pk_1,pk_2,pk_3,pk_4,pk_5])
    # Ensure the total length is k
    random_a_values = Pd[:k]

    print(random_a_values)
    # Define the stochastic matrix P(t) for a given time t
    def stochastic_matrix(t):
        return np.array([[Pd[t], 1 - Pd[t]        , 0    ],
                         [Pd[t], 1 - Pd[t] - Pk[t], Pk[t]],
                         [0, 1 - Pk[t], Pk[t]]
                         ])


    # Initial probability distribution at time t
    initial_distribution = np.array([0.9, 0.1, 0.])  # Example initial distribution

    # Perform matrix exponentiation for each time step
    result_matrix = np.identity(3)  # Identity matrix to start

    for t in range(k):
        result_matrix = result_matrix.dot(stochastic_matrix(t))
        if t > 39 and t< 60:
            print("t: ", t, " pd: ", Pd[t], " Pk: ", Pk[t])
            print("========================")
            # print(result_matrix)
            print(initial_distribution.dot(result_matrix))
            print()

    """
    S = {
    je skryt,
    není skryt,
    je kaput
    }
    přechod do je kaput - možná z predikovaných bboxu a poměru barev zda je stále překážka
    """
    # Calculate the final probability distribution at time t+k
    final_distribution = initial_distribution.dot(result_matrix)

    # Print the result matrix and final distribution
    print("Result Matrix:")
    print(result_matrix)

    print("\nFinal Probability Distribution:", final_distribution)
