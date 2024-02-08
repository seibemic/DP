import numpy as np
from scipy.stats import multivariate_normal as mvn
import cv2
from src.impl.MTT.confidence_ellipse import cv2_confidence_ellipse


class SpawnPoint:
    def __init__(self, m, cov, w):
        self.m = m
        self.cov = cov
        self.w = w
class TargetTracker:
    def __init__(self, F, H, Q, R, ps):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.ps = ps
        self.trackers = []
        self.spawnPoints = []
        self.imageSize = None
    def add_SpawnPoint(self, pos, w, cov):
        sp = SpawnPoint(pos, cov , w)
        self.spawnPoints.append(sp)

    def set_ImageSize(self, imageSize):
        self.imageSize = imageSize
    def add_PerimeterSpawnPoints(self, w, cov, n_std = 3):
        cov_2x2 = np.float32(cov[:2, :2])

        # Calculating eigenvalues
        eigenvalues = np.linalg.eigvals(cov_2x2)

        # Extracting major and minor axes lengths
        x_length = n_std * np.sqrt(eigenvalues[0])
        y_length = n_std * np.sqrt(eigenvalues[1])

        # Width_l
        x_num_points = (self.imageSize[0]) / x_length / 1.5

        # Height
        y_num_points = (self.imageSize[1]) / y_length / 1.5

        x_space_between = (self.imageSize[0]) / x_num_points
        y_space_between = (self.imageSize[1]) / y_num_points

        x_border_space = (self.imageSize[0] - x_space_between * int(x_num_points)) / 2
        y_border_space = (self.imageSize[1] - y_space_between * int(y_num_points)) / 2

        for i, row in enumerate(range(round(x_num_points))):
            x_pos = i * x_space_between + x_border_space
            y_pos = y_border_space
            pos = np.array([x_pos, y_pos])
            self.add_SpawnPoint(pos, w, cov)
            pos = np.array([x_pos, self.imageSize[1] - y_border_space])
            self.add_SpawnPoint(pos, w, cov)


        for i, row in enumerate(range(round(y_num_points) - 1)):
            x_pos = x_border_space
            y_pos = i * y_space_between + y_border_space + y_space_between
            pos = np.array([x_pos, y_pos])
            self.add_SpawnPoint(pos, w, cov)
            pos = np.array([self.imageSize[0] - x_border_space, y_pos])
            self.add_SpawnPoint(pos, w, cov)


    def add_MeshSpawnPoints(self, w, cov, n_std = 3):
        cov_2x2 = np.float32(cov[:2, :2])

        # Calculating eigenvalues
        eigenvalues = np.linalg.eigvals(cov_2x2)

        # Extracting major and minor axes lengths
        x_length = n_std * np.sqrt(eigenvalues[0])
        y_length = n_std * np.sqrt(eigenvalues[1])

        # Width_l
        x_num_points = (self.imageSize[0]) / x_length / 1.5

        # Height
        y_num_points = (self.imageSize[1]) / y_length / 1.5

        x_space_between = (self.imageSize[0]) / x_num_points
        y_space_between = (self.imageSize[1]) / y_num_points

        x_border_space = (self.imageSize[0] - x_space_between * int(x_num_points)) / 2
        y_border_space = (self.imageSize[1] - y_space_between * int(y_num_points)) / 2


        for j, _ in enumerate(range(round(y_num_points) + 1)):
            for i, __ in enumerate(range(round(x_num_points))):
                x_pos = i * x_space_between + x_border_space
                y_pos = j * y_space_between + y_border_space
                pos = np.array([x_pos, y_pos])
                self.add_SpawnPoint(pos, w, cov)

    def show_SpawnPoints(self, img, color = (255,0,0), thickness = 1):
        #print(self.imageSize)
        #print(len(self.spawnPoints))
        for sp in self.spawnPoints:
            #print(sp.m)
            img = cv2_confidence_ellipse(sp.m, sp.cov, img, 3, color,thickness)
        return img

