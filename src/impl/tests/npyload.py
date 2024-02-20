import numpy as np

if __name__ == "__main__":
    hist1 = np.load("/home/michal/Documents/FIT/DP/dp/src/impl/Video_MTT_Manager/hist1.npy")
    hist2 = np.load("/home/michal/Documents/FIT/DP/dp/src/impl/Video_MTT_Manager/hist2.npy")

    # print(hist1)
    cos_sim = np.zeros(shape=(hist1.shape[0]))
    for i, val in enumerate(hist1):
        cos_sim[i] = hist2[i] @ val / (np.linalg.norm(hist2[i]) * np.linalg.norm(val))
    print(cos_sim)