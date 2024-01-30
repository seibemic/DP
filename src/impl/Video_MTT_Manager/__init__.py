import torch
import warnings
import cv2
import os
import re
from src.impl.YOLO import YOLOHandler
from src.impl.SAM import SAM_handler


class VideoMTT:
    def __init__(self, input_video=None, MTT=None, SAM=None, YOLO=None, output_video=None):
        self.m_output_video = None
        self.m_YOLO = None
        self.m_SAM = None
        self.m_MTT = None
        self.m_input_video = None
        self.set_inputVideo(input_video)
        self.set_MTT(MTT)
        self.set_SAM(SAM)
        self.set_YOLO(YOLO)
        self.set_outputVideo(output_video)
        self.m_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def set_inputVideo(self, input_video):
        if not os.path.exists(input_video):
            raise Exception("Input file does not exists.")
        pattern = r".*.mp4"
        if not re.match(pattern, input_video):
            raise Exception("Input file is not mp4 format.")
        self.m_input_video = input_video

    def set_MTT(self, MTT):
        self.m_MTT = MTT

    def set_SAM(self, SAM):
        self.m_SAM = SAM

    def set_YOLO(self, YOLO):
        self.m_YOLO = YOLO

    def set_outputVideo(self, output_video):
        self.m_output_video = output_video

    def set_device(self, device):
        if device != "cpu" and not torch.cuda.is_available():
            warnings.warn("GPU is not available, setting device to cpu.")
            self.m_device = "cpu"
            return
        self.m_device = "cpu"
        return

    def checkClassMembers(self):
        if self.m_MTT == None:
            raise Exception("MTT is not set")
        if not isinstance(self.m_SAM, SAM_handler):
            raise Exception("SAM is not set")
        if not isinstance(self.m_YOLO, YOLOHandler):
            raise Exception("YOLO is not set")
        if self.m_output_video == None:
            raise Exception("Output video is not set")
        if self.m_input_video == None:
            raise Exception("Input video is not set")

    def get_videoDimensions(self, cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_videoFps(self, cap):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return fps

    def get_outputVideoWriter(self, input_cap, output_path):
        # Get the video's properties (width, height, FPS)
        width, height = self.get_videoDimensions(input_cap)
        fps = self.get_videoFps(input_cap)

        # Define the output video file
        output_codec = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        output_video = cv2.VideoWriter(output_path, output_codec, fps, (width, height))

        return output_video

    def get_color(self, color):
        return int(color[0]), int(color[1]), int(color[2])

    def add_color(self, mask, color):
        next_mask = mask.astype(np.uint8)
        next_mask = np.expand_dims(next_mask, 0).repeat(3, axis=0)
        next_mask = np.moveaxis(next_mask, 0, -1)
        return next_mask * color

    # Merge masks into a single, multi-colored mask
    def merge_masks_colored(self, masks, class_ids):
        filtered_class_ids = []
        filtered_masks = []
        for idx, cid in enumerate(class_ids):
            if int(cid) in self.m_YOLO.chosen_class_ids:
                filtered_class_ids.append(cid)
                filtered_masks.append(masks[idx])

        merged_with_colors = self.add_color(filtered_masks[0][0],
                                            self.get_color(self.m_YOLO.colors[int(filtered_class_ids[0])])).astype(
            np.uint8)

        if len(filtered_masks) == 1:
            return merged_with_colors

        for i in range(1, len(filtered_masks)):
            curr_mask_with_colors = self.add_color_to_mask(filtered_masks[i][0], self.get_color(
                self.m_YOLO.colors[int(filtered_class_ids[i])]))
            merged_with_colors = np.bitwise_or(merged_with_colors, curr_mask_with_colors)

        return merged_with_colors.astype(np.uint8)

    def run(self):
        self.checkClassMembers()
        videoCap = cv2.VideoCapture(self.m_input_video)
        output_video_boxes = self.get_outputVideoWriter(videoCap, self.m_output_video + "_boxes.mp4")
        output_video_masks = self.get_outputVideoWriter(videoCap, self.m_output_video + "_masks.mp4")
        # print(self.m_input_video)
        frame_num = 1
        while videoCap.isOpened():
            print("frame: ", frame_num)
            ret, frame = videoCap.read()
            if not ret:
                break
            yoloDetections = self.m_YOLO.predict(frame)
            frameWithYoloDetections = self.m_YOLO.visualizeDetectionsBbox(frame,
                                                                          yoloDetections[0].boxes.cpu().xyxy,
                                                                          yoloDetections[0].boxes.cpu().conf,
                                                                          yoloDetections[0].boxes.cpu().cls)
            output_video_boxes.write(frameWithYoloDetections)
            # output_video_boxes.write(frame)

            transformedBoxes = self.m_SAM.transformBoxes(detections=yoloDetections,
                                                         video_dims=list(self.get_videoDimensions(videoCap)))

            if len(transformedBoxes) == 0:
                print("No boxes found on frame", frame_num)
                output_video_masks.write(frame)
                frame_num += 1
                continue
            masks, scores, logits = self.m_SAM.predict(frame, transformedBoxes)
            if masks is None or len(masks) == 0:
                print("No masks found on frame", frame_num)
                output_video_masks.write(frame)
                frame_num += 1
                continue

            merged_colored_mask = self.merge_masks_colored(masks, yoloDetections[0].boxes.cls)

            # Write masks to output video
            image_combined = cv2.addWeighted(frame, 0.7, merged_colored_mask, 0.7, 0)
            output_video_masks.write(image_combined)

            frame_num += 1
            if frame_num > 60:
                break
        videoCap.release()
        output_video_boxes.release()
        output_video_masks.release()
        cv2.destroyAllWindows()
