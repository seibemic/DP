import numpy as np
import cv2
from scipy.special import kl_div
class ObjectStats:
    def __init__(self, frame, mask):
        self.frame = frame
        self.mask = mask
        self.hue, self.saturation, self.value = self.get_object_histogram(frame, mask)



    def get_object_histogram(self, rgb_image, binary_mask):
        # Apply the binary mask to the RGB image
        masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        # Calculate the histogram
        hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        return hist_hue[1:].flatten(), hist_saturation[1:].flatten(), hist_value[1:].flatten()

    def get_cosineSimilarity(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = self.hue @ hue / (np.linalg.norm(self.hue) * np.linalg.norm(hue))
        sat_cmp = self.saturation @ sat / (np.linalg.norm(self.saturation) * np.linalg.norm(sat))
        val_cmp = self.value @ val / (np.linalg.norm(self.value) * np.linalg.norm(val))
        if print_result:
            print("cosine similarity (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp
    def get_intersection(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = np.sum(np.minimum(self.hue, hue)) / np.sum(self.hue)
        sat_cmp = np.sum(np.minimum(self.saturation, sat)) / np.sum(self.saturation)
        val_cmp = np.sum(np.minimum(self.value, val)) / np.sum(self.value)
        if print_result:
            print("intersection (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_correlation(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = np.abs(cv2.compareHist(self.hue, hue, cv2.HISTCMP_CORREL))
        sat_cmp = np.abs(cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CORREL))
        val_cmp = np.abs(cv2.compareHist(self.value, val, cv2.HISTCMP_CORREL))
        if print_result:
            print("correlation (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp
    def get_chi(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_CHISQR) / np.sum(self.hue)
        sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CHISQR) / np.sum(self.saturation)
        val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_CHISQR) / np.sum(self.value)
        if print_result:
            print("chi square (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_bhat(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_BHATTACHARYYA)
        sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_BHATTACHARYYA)
        val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_BHATTACHARYYA)
        if print_result:
            print("bhat (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_hellinger(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_HELLINGER)
        sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_HELLINGER)
        val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_HELLINGER)
        if print_result:
            print("hellinger (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_chi_alt(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.hue)
        sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.saturation)
        val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.value)
        if print_result:
            print("chi alt (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_KLdiv(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_KL_DIV)
        sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_KL_DIV)
        val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_KL_DIV)
        if print_result:
            print("KL div (hue, sat, val):")
            print(hue_cmp, sat_cmp, val_cmp)
        return hue_cmp, sat_cmp, val_cmp

    def get_KLdiv2(self, mask, print_result=False):
        hue, sat, val = self.get_object_histogram(self.frame, mask)
        hue_cmp = kl_div(self.hue, hue)
        sat_cmp = kl_div(self.saturation, sat)
        val_cmp = kl_div(self.value, val)
        if print_result:
            print("KL div (hue, sat, val):")
            print(np.sum(hue_cmp), np.sum(sat_cmp), np.sum(val_cmp))
            print(np.sum(hue_cmp)/len(hue_cmp), np.sum(sat_cmp)/len(sat_cmp), np.sum(val_cmp)/len(val_cmp))
        return hue_cmp, sat_cmp, val_cmp

    def printAll(self, mask):
        self.get_cosineSimilarity(mask, True)
        self.get_intersection(mask, True)
        self.get_correlation(mask, True)
        self.get_chi(mask, True)
        self.get_bhat(mask, True)
        self.get_hellinger(mask, True)
        self.get_chi_alt(mask, True)
        self.get_KLdiv2(mask, True)


