import numpy as np
import sys
sys.path.append("../..")
# from src.impl.Frame_Processing.YOLO import YOLOHandler
# from src.impl.Frame_Processing.SAM import SAM_handler
from src.impl.Video_MTT_Manager import VideoMTT
from src.impl.MTT.PHD.PHD_tracker import PHDTracker
from src.impl.Frame_Processing import FrameProcessing
if __name__ == '__main__':
    dt = 1/25*25

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.diag([0.1, 0.1, 0.1, 0.1])
    R = np.diag([5, 5]) * 50
    H = np.diag([1, 1])  # 2x4
    H = np.lib.pad(H, ((0, 0), (0, 2)), 'constant', constant_values=(0))
    Ps = 0.95


    MTT = PHDTracker(F,H,Q,R,Ps)

    of = "/home/michal/Documents/FIT/DP/dp/src/data/output/test01"
    frameProcessor = FrameProcessing(mode=0, device="cpu")
    # yolo = YOLOHandler()

    # sam = SAM_handler(device = "cpu")
    input = "/home/michal/Documents/FIT/DP/dp/src/data/input/DSCN0005.MP4"
    vid = VideoMTT(input_video=input, MTT = MTT, frameProcessor=frameProcessor,  chosen_class_ids=[0], output_video=of)
    vid.run()
