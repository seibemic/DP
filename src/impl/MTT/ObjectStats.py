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
        self.maskValues = self.get_object_histogram(frame, mask)
        inverseMask = self.get_InverseMaskWithinBbox(self.xyxy)
        # inverseMask = self.get_xyxyMask(self.xyxy)
        self.inverseMaskValues = self.get_object_histogram(frame, inverseMask, all_spectrums=False)
        self.timestamp = timestamp

    def get_object_histogram(self, rgb_image, binary_mask, all_spectrums = True):
        masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)

        spectrums =[]
        if all_spectrums == False:
            rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            xyz_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2XYZ)
            spectrums.append(xyz_image)
            spectrums.append(rgb_image)
            all_arrs = []
            for j, spectrum in enumerate(spectrums):
                for i in range(spectrum.shape[2]):
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
                    if j == 3:
                        all_arrs.append(cv2.calcHist([spectrum], [0], None, [256], [0, 256])[1:].flatten())
                        break
                    all_arrs.append(cv2.calcHist([spectrum], [i], None, [256], [0, 256])[1:].flatten())

            all_arrs = np.array(all_arrs)
            e = 1e-10
            return all_arrs / (np.sum(all_arrs) + e)

    def get_cosineSimilarity(self, hist1, hist2, print_result=False):
        assert len(hist1) == len(hist2)
        cos_sim = np.zeros(shape=(hist1.shape[0]))
        np.save("hist1.npy", hist1)
        np.save("hist2.npy", hist2)
        th = 1e-20
        for i, val in enumerate(hist1):
            cos_sim[i] = (hist2[i]+th) @ (val+th) / (np.linalg.norm((hist2[i]+th)) * np.linalg.norm((val)+th))
        if print_result:
            print("cosine similarity:")
            print(cos_sim)
        return cos_sim

    def get_intersection(self, hist1, hist2, print_result=False):
        assert len(hist1) == len(hist2)
        inter = np.zeros(shape=(hist1.shape[0]))
        th = 1e-20
        for i, val in enumerate(hist1):
           # print("inter: ", np.sum(hist2[i]))
            #inter[i] = np.sum(np.minimum(hist2[i], val)) / np.sum(hist2[i])
            #inter[i] = np.sum(np.minimum(hist2[i], val)) / len(hist2[i])
            minima = np.minimum(hist2[i], val)
            inter[i] = np.true_divide(np.sum(minima), np.sum(hist2[i])+th)

        if print_result:
            print("intersection:")
            print(inter)
        return inter


    def get_correlation(self, hist1, hist2, print_result=False):
        assert len(hist1) == len(hist2)
        corr = np.zeros(shape=(hist1.shape[0]))
        for i, val in enumerate(hist1):
            corr[i] = np.abs(cv2.compareHist(hist2[i], val, cv2.HISTCMP_CORREL))

        if print_result:
            print("correlation:")
            print(corr)
        return corr


    def printAll(self, mask):
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

    def get_maskStatsMean(self, frame, mask):
        #hist1 = self.get_object_histogram(self.frame, mask)
        hist1 = self.get_object_histogram(frame, mask)
        hist2 = self.maskValues
        all_vals = []
        all_vals.append(self.get_cosineSimilarity(hist1, hist2, print_result=False))
        all_vals.append(self.get_intersection(hist1, hist2, print_result=False))
        all_vals.append(self.get_correlation(hist1, hist2, print_result=False))
        all_vals = np.array(all_vals)

        return np.mean(all_vals)

    def get_xyxyStatsMean(self, frame, xyxy):
        xyxy1 = self.get_xyxyMask(xyxy)
        hist1 = self.get_object_histogram(frame, xyxy1, all_spectrums=True)
        xyxy2 = self.get_xyxyMask(self.xyxy)
        hist2 = self.get_object_histogram(frame, xyxy2, all_spectrums=True)

        all_vals = []
        all_vals.append(self.get_cosineSimilarity(hist1, hist2, print_result=False))
        all_vals.append(self.get_intersection(hist1, hist2, print_result=False))
        all_vals.append(self.get_correlation(hist1, hist2, print_result=False))
        all_vals = np.array(all_vals)
        return np.mean(all_vals)


    def plot_histogram(self, hist, title, frame_num):
        plt.plot(hist)
        plt.title(title+str(frame_num))
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        # plt.savefig(f"/home/michal/Documents/FIT/DP/dp/src/data/output/graphs/"+title + f"_{frame_num}.png")
        # plt.show()
