import numpy as np
from scipy.stats import chi2
import cv2
from PIL import ImageChops, Image
import matplotlib.pyplot as plt
from src.impl.MTT.ObjectStats import ObjectStats
from src.impl.MTT.MarkovChain import MarkovChain
from copy import deepcopy
class PHD:
    def __init__(self, w, m, P, pd = 0.9, xyxy = None, mask = None, objectStats=None, markovChain=None, fromMerge=False):
        self.prev_m = None
        self.w = w
        self.m = m
        self.P = P
        self.pd = pd
        self.xyxy = xyxy
        self.mask = mask
        self.objectStats = objectStats
        self.state = 0
        if markovChain is None:
            init_dist = np.array([0.4,0.3,0.3])
            self.markovChain = MarkovChain(init_dist)
            self.state = np.argmax(self.markovChain.get_probs())
        else:
            if not fromMerge:
                self.markovChain = deepcopy(markovChain)
                pk = 1
                self.state = np.argmax(self.markovChain.get_transitionProbs(self.pd, pk))
            else:
                self.markovChain = deepcopy(markovChain)
                self.state = np.argmax(self.markovChain.get_probs())


        # self.P_aposterior=self.P_aprior


    def predict(self, ps, A, Q, frameShape):
        self.w = ps * self.w
        self.prev_m = self.m
        self.m = A @ self.m
        if self.xyxy is not None:
            x1, y1, x2, y2 = self.xyxy
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            frame_diag = np.sqrt(frameShape[0] ** 2 + frameShape[1] ** 2)
            Q *= frame_diag / length * 2
        self.P = Q + A @ self.P @ A.T


    def updateComponents(self, H, R):
        self.ny = H @ self.m
        self.S = R + H @ self.P @ H.T
        self.K = self.P @ H.T @ np.linalg.inv(self.S)
        self.P_apost = self.P.copy()
        self.P = (np.eye(len(self.K)) - self.K @ H) @ self.P

    def moveMask_and_getPd(self, frame, defaultPd = 0.3):
        if self.mask is not None:
            dx = self.m[2]
            dy = self.m[3]
            print("dx, dy: ", dx, dy)
            self.move_binary_mask(dx, dy)
            self.pd = self.getPd(frame)
            # print("pd: ", self.pd)
        else:
            self.pd = defaultPd
      #  self.pd = 0.95
    def moveBbox(self, t=1):
        self.xyxy = self.xyxy + t * np.array([self.m[2], self.m[3], self.m[2], self.m[3]])
    def update(self, H, pd, frame, frame_num):
        t = 2
        if self.xyxy is not None and self.objectStats is not None and (frame_num - self.objectStats.timestamp) % t == 1:
            self.moveBbox(t)
        pk = 1
        if self.objectStats is not None and self.xyxy is not None:
            pk = self.getPk(self.xyxy, frame)

        self.m = self.m
        self.P = self.P_apost

        x = self.markovChain.get_transitionProbs(self.pd, pk)

        self.state = np.argmax(x)
        self.w = (1 - self.pd) * self.w
    def getPd(self, frame):

        return self.objectStats.get_maskStatsMean(frame, self.mask)
        # self.getMaskStats(frame)

    def getPk(self,xyxy,frame):
        PK = self.objectStats.get_xyxyStatsMean(frame, xyxy)
        # print("PK: ", PK)
        # print("PD: ", self.pd)
        return PK
    def inGating(self, z, Pg=0.99):
        covInv = np.linalg.inv(self.S)
        gamma = chi2.ppf(Pg, df=2)
        if ((z - self.ny).T @ covInv @ (z - self.ny)) <= gamma:
            return True
        return False
    def move_mask(self, dx, dy):
        rows, cols = self.mask.shape

        # Create an empty mask of the same size
        # moved_mask = np.zeros_like(self.mask)

        # Define the transformation matrix for translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the translation to the mask
        moved_mask = cv2.warpAffine(self.mask, M, (cols, rows))
        self.mask = moved_mask
        # return moved_mask

    def move_binary_mask(self, dx, dy):
        # Ensure dx and dy are integers
        dx = int(dx)
        dy = int(dy)

        # Use binary shifting to move the object in x direction
        if dx > 0:
            self.mask[:, dx:] = self.mask[:, :-dx]
            self.mask[:, :dx] = 0
        elif dx < 0:
            self.mask[:, :dx] = self.mask[:, -dx:]
            self.mask[:, dx:] = 0

        # Use binary shifting to move the object in y direction
        if dy > 0:
            self.mask[dy:, :] = self.mask[:-dy, :]
            self.mask[:dy, :] = 0
        elif dy < 0:
            self.mask[:dy, :] = self.mask[-dy:, :]
            self.mask[dy:, :] = 0

    def first_nonzero_index(self, arr):
        indices = np.nonzero(arr)
        if indices[0].size == 0:  # Check if the array is entirely zero
            return None
        return (indices[0][0], indices[1][0])
