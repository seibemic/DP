import numpy as np
from src.impl.YOLO import YOLOHandler
from src.impl.SAM import SAM_handler
from src.impl.Video_MTT_Manager import VideoMTT
if __name__ == '__main__':
    of = "/home/michal/Documents/FIT/DP/dp/src/data/output/skateboarding"
    yolo = YOLOHandler()
    mtt = "XXX"
    sam = SAM_handler(device = "cpu")
    input = "/home/michal/Documents/FIT/DP/dp/src/data/input/skateboarding.mp4"
    vid = VideoMTT(input_video=input, MTT = mtt, SAM=sam, YOLO=yolo, output_video=of)
    vid.run()
