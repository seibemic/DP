import numpy as np

# Define the time-dependent function a(t)
T= np.random.rand(1000)
T= np.array([0.9,0.9])
def a(t):
    # Your implementation of a(t) goes here
    return T[t]  # Replace this with your actual function

# Define the stochastic matrix P(t) for a given time t
def stochastic_matrix(t):
    return np.array([[1 - a(t), a(t)], [a(t), 1 - a(t)]])

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
    random_a_values = np.concatenate([high_values_1, low_values_1, high_values_2,low_values_2, high_values_3])

    # Ensure the total length is k
    random_a_values = random_a_values[:k]

    print(random_a_values)
    # Define the stochastic matrix P(t) for a given time t
    def stochastic_matrix(t):
        return np.array([[random_a_values[t], 1 - random_a_values[t]], [random_a_values[t], 1 - random_a_values[t]]])


    # Initial probability distribution at time t
    initial_distribution = np.array([0.9, 0.1])  # Example initial distribution

    # Perform matrix exponentiation for each time step
    result_matrix = np.identity(2)  # Identity matrix to start

    for t in range(k):
        result_matrix = result_matrix.dot(stochastic_matrix(t))
        if t > 39 and t< 60:
            print("t: ", t, " pd: ", random_a_values[t])
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
