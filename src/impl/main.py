import numpy as np
import torch
import sys
sys.path.append("../..")
from src.impl.Video_MTT_Manager import VideoMTT
from src.impl.MTT.PHD.PHD_tracker import PHDTracker
from src.impl.Frame_Processing import FrameProcessing
if __name__ == '__main__':
    dt = 1/25*25

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1.06, 0],
                  [0, 0, 0, 1.06]])
    Q = np.diag([1., 1., 1., 1.])*0.04
    Q = np.array([[2, 0.5, 0, 0],
                  [0.5, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]) * 0.004
    R = np.diag([1, 1]) * 120#60#100#80
    R= np.array([[2,0.5],
                 [0.5,1]])*20.
    H = np.diag([1, 1])  # 2x4
    H = np.lib.pad(H, ((0, 0), (0, 2)), 'constant', constant_values=(0))
    Ps = 0.99

    MTT = PHDTracker(F,H,Q,R,Ps)

    of = "../data/output/test01" # output file
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #   modes: 0 - S1 - YOLO + YOLO
    #          1 - S2 - YOLO + SAM
    #          2 - S3 - Grounded SAM
    frameProcessor = FrameProcessing(mode=0, device=device)

    #VID20240229170542.mp4
    # VID20240229170959.mp4
    # yt_traffic01.mp4 https://www.youtube.com/watch?v=KBsqQez-O4w&t=30s&ab_channel=NickMartinez
    # yt_traffic03.mp4 https://www.youtube.com/watch?v=7WFYiZersNc&ab_channel=AbdulMunaim
    # yt_traffic04.mp4 https://www.youtube.com/watch?v=ddPnEk90vLk&ab_channel=IMFootage
    input = "../data/input/yt_traffic03.mp4" # input file
    vid = VideoMTT(input_video=input, MTT = MTT, frameProcessor=frameProcessor,  chosen_class_ids=[2], output_video=of)

    d = 40
    P = np.array([[2*d, d/2, 0, 0],
                  [d/2, d, 0, 0],
                  [0, 0, d, 0],
                  [0, 0, 0, d]])
    vid.run(P)
