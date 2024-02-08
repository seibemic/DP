import numpy as np
from scipy.stats import chi2

class PHD:
    def __init__(self, w, m, P, conf = 0.9, xyxy = None, prev_xyxy=None, mask = None):
        self.prev_m = None
        self.w = w
        self.m = m
        self.P = P
        self.conf = conf
        self.xyxy = xyxy
        self.prev_xyxy = prev_xyxy
        self.mask = mask
        # self.P_aposterior=self.P_aprior


    def predict(self, ps, A, Q):
        self.w = ps * self.w
        self.prev_m = self.m
        self.m = A @ self.m
        self.P = Q + A @ self.P @ A.T


    def updateComponents(self, H, R):
        self.ny = H @ self.m
        self.S = R + H @ self.P @ H.T
        self.K = self.P @ H.T @ np.linalg.inv(self.S)
        self.P_apost = self.P.copy()
        self.P = (np.eye(len(self.K)) - self.K @ H) @ self.P

    def update(self, H, conf = 0.9):
        self.w = (1 - self.conf) * self.w
        # self.w = (1 - 0.3) * self.w
        self.m = self.m
        self.P = self.P_apost
        self.prev_xyxy = self.xyxy
        if self.xyxy is not None:
            self.xyxy = self.xyxy + np.tile(H @ (self.m - self.prev_m) , 2)
        # self.P_aposterior = self.P_aprior

    def inGating(self, z, Pg=0.99):
        covInv = np.linalg.inv(self.S)
        gamma = chi2.ppf(Pg, df=2)
        if ((z - self.ny).T @ covInv @ (z - self.ny)) <= gamma:
            return True
        return False
