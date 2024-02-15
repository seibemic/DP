from src.impl.Frame_Processing.Stream import Stream
import cv2

if __name__ == "__main__":
    # stream = Stream(0, 5) # subsample your webcam from probably 30fps to 5fps
    stream = Stream("/home/michal/Documents/FIT/DP/dp/src/data/input/DSCN0002.MP4", 5) # will take on frame over 6 from your video

    frame_num = 0
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            frame_num +=1
            print(frame_num)
            cv2.namedWindow(f"{frame_num}", cv2.WINDOW_NORMAL)
            # Using resizeWindow()
            cv2.resizeWindow(f"{frame_num}", 1600, 1200)

            cv2.imshow(f"{frame_num}", frame)
            cv2.waitKey(0)
    print(frame_num)