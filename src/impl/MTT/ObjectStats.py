import numpy as np
import cv2
from scipy.special import kl_div
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency


class ObjectStats:
    def __init__(self, frame, mask, xyxy, timestamp):
        self.frame = frame
        self.mask = mask
        self.xyxy = xyxy.astype(int)
        # self.hue, self.saturation, self.value = self.get_object_histogram(frame, mask)
        self.maskValues = self.get_object_histogram(frame, mask)
        inverseMask = self.get_InverseMaskWithinBbox(self.xyxy)
        # inverseMask = self.get_xyxyMask(self.xyxy)
        self.inverseMaskValues = self.get_object_histogram(frame, inverseMask, all_spectrums=False)
        self.timestamp = timestamp

    def get_object_histogram(self, rgb_image, binary_mask, all_spectrums = True):
        # Apply the binary mask to the RGB image
        masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)

        # Convert the image to HSV color space
        spectrums =[]
        if all_spectrums == False:
            rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            xyz_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2XYZ)
            spectrums.append(xyz_image)
            spectrums.append(rgb_image)
            all_arrs = []
            for j, spectrum in enumerate(spectrums):
                for i in range(spectrum.shape[2]):
                    # print("spectrum shape: ", spectrum.shape)
                    all_arrs.append(cv2.calcHist([spectrum], [i], None, [256], [0, 256])[1:].flatten())

            all_arrs = np.array(all_arrs)
            return all_arrs / np.sum(all_arrs)
        else:
            rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            xyz_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2XYZ)
            spectrums.append(xyz_image)
            spectrums.append(rgb_image)
            hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
            lab_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
            hls_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HLS)
            spectrums.append(hsv_image)
            spectrums.append(lab_image)
            spectrums.append(hls_image)
            all_arrs = []
            for j, spectrum in enumerate(spectrums):
                for i in range(spectrum.shape[2]):
                    if j == 1:
                        all_arrs.append(cv2.calcHist([spectrum], [0], None, [256], [0, 256])[1:].flatten())
                        break
                    # print("spectrum shape: ", spectrum.shape)
                    all_arrs.append(cv2.calcHist([spectrum], [i], None, [256], [0, 256])[1:].flatten())

            all_arrs = np.array(all_arrs)
            return all_arrs / np.sum(all_arrs)

    def get_cosineSimilarity(self, mask, object, print_result=False):

        if object == "mask":
            maskValues = self.maskValues
            vals = self.get_object_histogram(self.frame, mask)
        elif object == "xyxy":
            maskValues = self.inverseMaskValues
            vals = self.get_object_histogram(self.frame, mask, all_spectrums=False)
        cos_sim = np.zeros(shape=(vals.shape[0]))
        # print("cos_sim shape: ", cos_sim.shape)
        # print("values shape: ", self.values.shape)
        # print("values[0] shape: ", self.values[0].shape)
        for i, val in enumerate(vals):
            cos_sim[i] = maskValues[i] @ val / (np.linalg.norm(maskValues[i]) * np.linalg.norm(val))
        if print_result:
            print("cosine similarity:")
            print(cos_sim)
        return cos_sim

    def get_intersection(self, mask, object, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        if object == "mask":
            maskValues = self.maskValues
        elif object == "xyxy":
            maskValues = self.inverseMaskValues
        inter = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            inter[i] = np.sum(np.minimum(maskValues[i], val)) / np.sum(maskValues[i])
        if print_result:
            print("intersection:")
            print(inter)
        return inter

    def get_correlation(self, mask, object, print_result=False):

        if object == "mask":
            maskValues = self.maskValues
            vals = self.get_object_histogram(self.frame, mask)
        elif object == "xyxy":
            maskValues = self.inverseMaskValues
            vals = self.get_object_histogram(self.frame, mask, all_spectrums=False)
        corr = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            corr[i] = np.abs(cv2.compareHist(maskValues[i], val, cv2.HISTCMP_CORREL))
        if print_result:
            print("correlation:")
            print(corr)
        return corr

    def printAll(self, mask, frame_num=0):
        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # self.plot_histogram(self.hue, "hue orig", frame_num)
        # self.plot_histogram(self.saturation, "saturation orig", frame_num)
        # self.plot_histogram(self.value, "value orig", frame_num)
        #
        # self.plot_histogram(hue, "hue new", frame_num)
        # self.plot_histogram(sat, "saturation new", frame_num)
        # self.plot_histogram(val, "value new", frame_num)

        self.get_cosineSimilarity(mask, True)
        self.get_intersection(mask, True)
        self.get_correlation(mask, True)


    def get_InverseMaskWithinBbox(self, xyxy):
        # Create a blank mask of the same shape as the input binary mask
        inverse_mask = np.zeros_like(self.mask)

        # Get the coordinates of the bounding box in xyxy format
        x1, y1, x2, y2 = xyxy

        # Create a rectangle in the inverse mask within the bounding box
        inverse_mask[y1:y2, x1:x2] = 1

        # Subtract the object mask from the inverse mask
        inverse_mask -= self.mask

        # Clip values to ensure that the mask only contains 0 or 1
        inverse_mask = np.clip(inverse_mask, 0, 1)

        return inverse_mask

    def get_xyxyMask(self, xyxy):
        xyxy_mask = np.zeros_like(self.mask)
        x1, y1, x2, y2 = xyxy.astype(int)
        xyxy_mask[y1:y2, x1:x2] = 1

        return xyxy_mask

    def get_StatsMean(self, mask, object):
        all_vals = []
        if object == "xyxy":
            print("xyxy self: ", self.xyxy)
            all_vals.append(self.get_cosineSimilarity(mask, object, print_result=True))
            all_vals.append(self.get_correlation(mask, object, print_result=True))
        else:
            all_vals.append(self.get_cosineSimilarity(mask, object, print_result=False))
            all_vals.append(self.get_intersection(mask, object, print_result=False))
            all_vals.append(self.get_correlation(mask, object, print_result=False))
        all_vals = np.array(all_vals)

        return np.mean(all_vals)
    def plot_histogram(self, hist, title, frame_num):
        plt.plot(hist)
        plt.title(title+str(frame_num))
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        # plt.savefig(f"/home/michal/Documents/FIT/DP/dp/src/data/output/graphs/"+title + f"_{frame_num}.png")
        # plt.show()
