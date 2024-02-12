import numpy as np
import cv2
from scipy.special import kl_div
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency


class ObjectStats:
    def __init__(self, frame, mask):
        self.frame = frame
        self.mask = mask
        # self.hue, self.saturation, self.value = self.get_object_histogram(frame, mask)
        self.values = self.get_object_histogram(frame, mask)


    def get_object_histogram(self, rgb_image, binary_mask):
        # Apply the binary mask to the RGB image
        masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)

        # Convert the image to HSV color space
        all_spectrums =[]
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        lab_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        cv2.COLOR_BGR2
        all_spectrums.append(hsv_image)
        all_spectrums.append(lab_image)
        all_spectrums.append(rgb_image)
        # hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2Lab)
        # hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
        # Calculate the histogram
        # print("hsv shape: ", hsv_image.shape)
        all_arrs =[]
        for spectrum in all_spectrums:
            for i in range(spectrum.shape[2]):
                # print("spectrum shape: ", spectrum.shape)
                all_arrs.append(cv2.calcHist([spectrum], [i], None, [256], [0, 256])[1:].flatten())

        all_arrs = np.array(all_arrs)
        # print("all arrs shape: ", all_arrs.shape)
        # hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        # hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        # hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        # hist_lum = cv2.calcHist([lab_image], [0], None, [256], [0, 256])
        return all_arrs
       # return np.array([hist_hue[1:].flatten(), hist_saturation[1:].flatten(), hist_value[1:].flatten(), hist_lum[1:].flatten()])

    # def get_object_histogram_rgb(self, rgb_image, binary_mask):
    #     rgb_image=rgb_image.transpose(2,0,1)
    #     red = rgb_image[0]*binary_mask
    #     green = rgb_image[1] * binary_mask
    #     blue = rgb_image[2] * binary_mask
    #
    #     hist_red = np.histogram(red, bins=256, range=(0, 256))
    #     hist_green = np.histogram(green, bins=256, range=(0, 256))
    #     hist_blue = np.histogram(blue, bins=256, range=(0, 256))
    #     # print("hist_red shape: ", hist_red[0][1:])
    #     # print("red shape: ", red.shape)
    #     return hist_red[0][1:], hist_green[0][1:], hist_blue[0][1:]
    #     # print("rgb shape: ", rgb_image.shape)
    #     # print("mask shape: ", binary_mask.shape)
    #     # print("rgb[0]: ", rgb_image[0].shape)
    #     # expanded_mask = np.expand_dims(binary_mask, axis=-1)
    #     # print("expanded mask: ", expanded_mask.shape)
    #     #
    #     # object_colors_red = rgb_image[expanded_mask]
    #     # print("obj shape: ",object_colors_red.shape)
    #     #return object_colors[0], object_colors[1], object_colors[2]
    def get_cosineSimilarity(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        cos_sim = np.zeros(shape=(vals.shape[0]))
        # print("cos_sim shape: ", cos_sim.shape)
        # print("values shape: ", self.values.shape)
        # print("values[0] shape: ", self.values[0].shape)
        for i, val in enumerate(vals):
            cos_sim[i] = self.values[i] @ val / (np.linalg.norm(self.values[i]) * np.linalg.norm(val))
        if print_result:
            print("cosine similarity:")
            print(cos_sim)
        return cos_sim


        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = self.hue @ hue / (np.linalg.norm(self.hue) * np.linalg.norm(hue))
        # sat_cmp = self.saturation @ sat / (np.linalg.norm(self.saturation) * np.linalg.norm(sat))
        # val_cmp = self.value @ val / (np.linalg.norm(self.value) * np.linalg.norm(val))
        # if print_result:
        #     print("cosine similarity (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp
    def get_intersection(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        inter = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            inter[i] = np.sum(np.minimum(self.values[i], val)) / np.sum(self.values[i])
        if print_result:
            print("intersection:")
            print(inter)
        return inter


        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = np.sum(np.minimum(self.hue, hue)) / np.sum(self.hue)
        # sat_cmp = np.sum(np.minimum(self.saturation, sat)) / np.sum(self.saturation)
        # val_cmp = np.sum(np.minimum(self.value, val)) / np.sum(self.value)
        # if print_result:
        #     print("intersection (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

    def get_correlation(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        corr = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            corr[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_CORREL))
        if print_result:
            print("correlation:")
            print(corr)
        return corr




        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = np.abs(cv2.compareHist(self.hue, hue, cv2.HISTCMP_CORREL))
        # sat_cmp = np.abs(cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CORREL))
        # val_cmp = np.abs(cv2.compareHist(self.value, val, cv2.HISTCMP_CORREL))
        # if print_result:
        #     print("correlation (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

    # def get_correlation_rgb(self, mask, print_result=False):

    def get_chi(self, mask, print_result=False, frame_num=0):
        vals = self.get_object_histogram(self.frame, mask)
        chi = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            chi[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_CHISQR))
        if print_result:
            print("chi:")
            print(chi)
        return chi

        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_CHISQR) / np.sum(self.hue)
        # sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CHISQR) / np.sum(self.saturation)
        # val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_CHISQR) / np.sum(self.value)

        # print(np.sum(self.hue/np.sum(self.hue)))
        # print(np.sum(hue/np.sum(hue)))
        # hue_chi =chisquare(self.hue/np.sum(self.hue), hue/np.sum(hue), ddof=1).pvalue
        # sat_chi =chisquare(self.saturation/np.sum(self.saturation), sat/np.sum(sat), ddof=1).pvalue
        # val_chi =chisquare(self.value/np.sum(self.value), val/np.sum(val), ddof=1).pvalue
        # normalized_hist1 = self.hue / np.sum(self.hue)
        # normalized_hist2 = hue / np.sum(hue)
        # chi2_stat, p_value = chisquare(normalized_hist1, f_exp=normalized_hist2)
        # epsilon = 1e-10
        #
        # observed_data_hue = np.array([self.hue+ epsilon, hue+ epsilon])
        # chi2_stat, p_value_hue, dof, expected = chi2_contingency(observed_data_hue)
        #
        # observed_data_sat= np.array([self.saturation+ epsilon, sat+ epsilon])
        # chi2_stat, p_value_sat, dof, expected = chi2_contingency(observed_data_sat)
        #
        # observed_data_val = np.array([self.value+ epsilon, val+ epsilon])
        # chi2_stat, p_value_val, dof, expected = chi2_contingency(observed_data_val)
        # np.save(f"/home/michal/Documents/FIT/DP/dp/src/data/output/npy/self_hue_{frame_num}.npy", self.hue)
        # np.save(f"/home/michal/Documents/FIT/DP/dp/src/data/output/npy/hue_{frame_num}.npy", hue)
        #
        # print(self.hue, hue)
        # if print_result:
        #     print("chi square (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        #     print("chi square pval (hue, sat, val): ")
        #     print(p_value_hue, p_value_sat,p_value_val)
        # return hue_cmp, sat_cmp, val_cmp

    def get_bhat(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        bhat = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            bhat[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_BHATTACHARYYA))
        if print_result:
            print("bhat:")
            print(bhat)
        return bhat

        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_BHATTACHARYYA)
        # sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_BHATTACHARYYA)
        # val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_BHATTACHARYYA)
        # if print_result:
        #     print("bhat (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

    def get_hellinger(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        hell = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            hell[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_HELLINGER))
        if print_result:
            print("hell:")
            print(hell)
        return hell

        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_HELLINGER)
        # sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_HELLINGER)
        # val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_HELLINGER)
        # if print_result:
        #     print("hellinger (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

    def get_chi_alt(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        chi_alt = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            chi_alt[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_CHISQR_ALT))
        if print_result:
            print("chi_alt:")
            print(chi_alt)
        return chi_alt


        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.hue)
        # sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.saturation)
        # val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_CHISQR_ALT) / np.sum(self.value)
        # if print_result:
        #     print("chi alt (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

    def get_KLdiv(self, mask, print_result=False):
        vals = self.get_object_histogram(self.frame, mask)
        kldiv = np.zeros(shape=(vals.shape[0]))
        for i, val in enumerate(vals):
            kldiv[i] = np.abs(cv2.compareHist(self.values[i], val, cv2.HISTCMP_KL_DIV))
        if print_result:
            print("kldiv:")
            print(kldiv)
        return kldiv

        # hue, sat, val = self.get_object_histogram(self.frame, mask)
        # hue_cmp = cv2.compareHist(self.hue, hue, cv2.HISTCMP_KL_DIV)
        # sat_cmp = cv2.compareHist(self.saturation, sat, cv2.HISTCMP_KL_DIV)
        # val_cmp = cv2.compareHist(self.value, val, cv2.HISTCMP_KL_DIV)
        # if print_result:
        #     print("KL div (hue, sat, val):")
        #     print(hue_cmp, sat_cmp, val_cmp)
        # return hue_cmp, sat_cmp, val_cmp

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
        self.get_chi(mask, True, frame_num)
        self.get_bhat(mask, True)
        self.get_hellinger(mask, True)
        self.get_chi_alt(mask, True)
        self.get_KLdiv(mask, True)

    def plot_histogram(self, hist, title, frame_num):
        plt.plot(hist)
        plt.title(title+str(frame_num))
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        # plt.savefig(f"/home/michal/Documents/FIT/DP/dp/src/data/output/graphs/"+title + f"_{frame_num}.png")
        # plt.show()
