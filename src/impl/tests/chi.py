import numpy as np
from scipy.stats import chisquare

if __name__ == '__main__':
    f_exp = np.array([44, 24, 29, 3]) / 100 * 189
    f_obs = np.array([43, 52, 54, 40])
    print(f_exp.sum())
    print(f_obs.sum())
    chi = chisquare(f_obs=f_obs, f_exp=f_exp)
    print(chi)