import numpy as np
from scipy.stats import chi2
import cv2
from PIL import ImageChops, Image
import matplotlib.pyplot as plt
from src.impl.MTT.ObjectStats import ObjectStats
class PHD:
    def __init__(self, w, m, P, conf = 0.9, xyxy = None, prev_xyxy=None, mask = None, objectStats=None):
        self.prev_m = None
        self.w = w
        self.m = m
        self.P = P
        self.conf = conf
        self.xyxy = xyxy
        self.prev_xyxy = prev_xyxy
        self.mask = mask
        self.prev_mask = None
        self.objectStats = objectStats
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

    def update(self, H, pd, frame, frame_num):

        self.w = (1 - pd) * self.w
        # self.w = (1 - self.conf) * self.w
        # self.w = (1 - 0.3) * self.w
        self.m = self.m
        self.P = self.P_apost
        self.prev_xyxy = self.xyxy
        if self.xyxy is not None:
            self.xyxy = self.xyxy + np.tile(H @ (self.m - self.prev_m) , 2)
        if self.mask is not None:
            self.prev_mask = self.mask.copy()
            # m = H @ (self.m - self.prev_m)
            # dx = m[0]
            # dy = m[1]
            dx = self.m[2]
            dy = self.m[3]
            print("dx, dy: ", dx, dy)
            self.move_binary_mask(dx, dy)
            print("prev mask sum: ", np.sum(self.prev_mask))
            print("   first non zero: ", self.first_nonzero_index(self.objectStats.mask))
            print("mask sum: ", np.sum(self.mask))
            print("   first non zero: ", self.first_nonzero_index(self.mask))
            print("w: ", self.w)
            if dx<0:
                self.objectStats.printAll(self.mask, frame_num)
            # self.getPd(frame)
        else:
            self.w = 0
            # self.w = 1
        # self.P_aposterior = self.P_aprior

    def getPd(self, frame):
        self.getMaskStats(frame)
    def getMaskStats(self, frame):
        new_frame = np.ma.array(frame[:, :, 0], mask=np.invert(self.mask))
        prev_frame = np.ma.array(frame[:, :, 0], mask=np.invert(self.prev_mask))
        hist_hue_new, hist_saturation_new, hist_value_new = self.get_object_histogram(frame, self.mask)
        # print(hist_hue, hist_saturation, hist_value)
        # self.plot_histogram(hist_hue_new[1:], 'Hue Histogram_next')
        # self.plot_histogram(hist_saturation_new[1:], 'Saturation Histogram_next')
        # self.plot_histogram(hist_value_new[1:], 'Value Histogram_next')
        hist_hue_prev, hist_saturation_prev, hist_value_prev = self.get_object_histogram(frame, self.prev_mask)
        # print(hist_hue, hist_saturation, hist_value)
        # print(hist_hue_prev[1:].shape, hist_hue_new[1:].shape)
        # print(hist_hue_prev[1:])
        # print(hist_hue_prev[1:].flatten())
        # self.plot_histogram(hist_hue_prev[1:], 'Hue Histogram_prev')
        # self.plot_histogram(hist_saturation_prev[1:], 'Saturation Histogram_prev')
        # self.plot_histogram(hist_value_prev[1:], 'Value Histogram_prev')
        cos_sim_hue = hist_hue_new[1:].flatten() @ hist_hue_prev[1:].flatten() / (
            np.linalg.norm(hist_hue_new[1:].flatten()) * np.linalg.norm(hist_hue_prev[1:].flatten()))

        cos_sim_sat = hist_saturation_new[1:].flatten() @ hist_saturation_prev[1:].flatten() / (
            np.linalg.norm((hist_saturation_new[1:].flatten()) * np.linalg.norm(hist_saturation_prev[1:].flatten())))

        cos_sim_val = hist_value_new[1:].flatten() @ hist_value_prev[1:].flatten() / (
            np.linalg.norm((hist_value_new[1:].flatten()) * np.linalg.norm(hist_value_prev[1:].flatten())))
        print("hue: ", cos_sim_hue)
        print("saturation: ", cos_sim_sat)
        print("value: ", cos_sim_val)

        hue_intersection = np.sum(np.minimum(hist_hue_new[1:].flatten(), hist_hue_prev[1:].flatten()))
        sat_intersection = np.sum(np.minimum(hist_saturation_new[1:].flatten(), hist_saturation_prev[1:].flatten()))
        val_intersection = np.sum(np.minimum(hist_value_new[1:].flatten(), hist_value_prev[1:].flatten()))
        print("hue intersection: ", hue_intersection / np.sum(hist_hue_new[1:].flatten()))
        print("sat_intersection: ", sat_intersection / np.sum(hist_saturation_new[1:].flatten()))
        print("val_intersection: ", val_intersection / np.sum(hist_value_new[1:].flatten()))

        hue_corr = cv2.compareHist(hist_hue_new[1:].flatten(), hist_hue_prev[1:].flatten(), cv2.HISTCMP_CORREL)
        sat_corr = cv2.compareHist(hist_saturation_new[1:].flatten(), hist_saturation_prev[1:].flatten(), cv2.HISTCMP_CORREL)
        val_corr = cv2.compareHist(hist_value_new[1:].flatten(), hist_value_prev[1:].flatten(), cv2.HISTCMP_CORREL)
        print("hue_corr: ", hue_corr)
        print("sat_corr: ", sat_corr)
        print("val_corr: ", val_corr)

        hue_chi = cv2.compareHist(hist_hue_new[1:].flatten(), hist_hue_prev[1:].flatten(), cv2.HISTCMP_CHISQR)
        sat_chi = cv2.compareHist(hist_saturation_new[1:].flatten(), hist_saturation_prev[1:].flatten(),
                                   cv2.HISTCMP_CHISQR)
        val_chi = cv2.compareHist(hist_value_new[1:].flatten(), hist_value_prev[1:].flatten(), cv2.HISTCMP_CHISQR)
        print("hue_chi: ", hue_chi/ np.sum(hist_hue_new[1:].flatten()))
        print("sat_chi: ", sat_chi/ np.sum(hist_saturation_new[1:].flatten()))
        print("val_chi: ", val_chi/ np.sum(hist_value_new[1:].flatten()))

        hue_bhat = cv2.compareHist(hist_hue_new[1:].flatten(), hist_hue_prev[1:].flatten(), cv2.HISTCMP_BHATTACHARYYA)
        sat_bhat = cv2.compareHist(hist_saturation_new[1:].flatten(), hist_saturation_prev[1:].flatten(),
                                  cv2.HISTCMP_BHATTACHARYYA)
        val_bhat = cv2.compareHist(hist_value_new[1:].flatten(), hist_value_prev[1:].flatten(), cv2.HISTCMP_BHATTACHARYYA)
        print("hue_bhat: ", hue_bhat)
        print("sat_bhat: ", sat_bhat)
        print("val_bhat: ", val_bhat)
        # prev_img = Image.fromarray(prev_frame.astype(np.uint8))
        # new_img = Image.fromarray(new_frame.astype(np.uint8))
        # diff = ImageChops.difference(prev_img, new_img).histogram()
        # print(diff)
        # min
        # max
        # mean
        # hist

    def plot_histogram(self, hist, title):
        plt.plot(hist)
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    def get_object_histogram(self, rgb_image, binary_mask):
        # Apply the binary mask to the RGB image
        masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        # Calculate the histogram
        hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        return hist_hue, hist_saturation, hist_value
    def updateWeight(self, pd):
        self.w = (1 - pd) * self.w

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

    def move_binary_mask2(self, dx, dy):
        # Find the indices of the non-zero elements in the binary mask
        binary_mask = self.mask
        indices = np.argwhere(binary_mask == 1)

        # Update the indices based on the displacement
        new_indices = indices + np.array([dy, dx])

        # Create a new binary mask with the moved elements
        new_binary_mask = np.zeros_like(binary_mask)

        # Ensure that the new indices are within the bounds of the array
        valid_indices = (new_indices[:, 0] >= 0) & (new_indices[:, 0] < binary_mask.shape[0]) & \
                        (new_indices[:, 1] >= 0) & (new_indices[:, 1] < binary_mask.shape[1])

        # Update the new binary mask with the valid moved indices
        new_indices = new_indices[valid_indices]
        new_binary_mask[new_indices[:, 0], new_indices[:, 1]] = 1
        self.mask = new_binary_mask
        # return new_binary_mask

    def first_nonzero_index(self, arr):
        indices = np.nonzero(arr)
        if indices[0].size == 0:  # Check if the array is entirely zero
            return None
        return (indices[0][0], indices[1][0])
