import numpy as np
from scipy.stats import chi2_contingency, chi2
import matplotlib.pyplot as plt
if __name__ == "__main__":
 # a =31,47,48,49,50,51
    n= 31
    hist1 = np.load(f"/home/michal/Documents/FIT/DP/dp/src/data/output/npy/self_hue_{n}.npy")
    hist2 = np.load(f"/home/michal/Documents/FIT/DP/dp/src/data/output/npy/hue_{n}.npy")
    epsilon = 1.1
    hist1 = (hist1 + epsilon) #/ np.sum((hist1 + epsilon))
    hist2 = (hist2 + epsilon) #/ np.sum((hist2 + epsilon))
    # print(hist1)
    # print(hist2)
    plt.plot(hist1, label="hist1")
    plt.plot(hist2, label="hist2")
    plt.legend()
    plt.title(f"{n}")
    plt.show()
    observed_data = np.array([hist1, hist2])

    chi2_stat, p_value, dof, expected = chi2_contingency(observed_data)

    # Print the results
    alpha = 0.01
    chi_2_critical = chi2.isf(alpha, dof)  # inverse survival function
    print("Critical statistic value:\t", round(chi_2_critical, 4))
    print(f"Chi-square test statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of freedom: {dof}")
    # print("Expected frequencies:")
    # print(expected)