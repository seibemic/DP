import torch
import warnings
import cv2
import os
import re
import numpy as np

from src.impl.Frame_Processing import FrameProcessing
from src.impl.MTT import TargetTracker
from src.impl.MTT.confidence_ellipse import cv2_confidence_ellipse
from src.impl.Frame_Processing.Stream import Stream
class VideoMTT:
    def __init__(self, input_video=None, MTT=None, frameProcessor=None, output_video=None, chosen_class_ids=None):
        self.frameShape = None
        self.output_video = None
        self.frameProcessor = frameProcessor
        self.MTT = None
        self.input_video = None
        self.set_inputVideo(input_video)
        self.set_MTT(MTT)
        self.set_outputVideo(output_video)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if chosen_class_ids is None:
            chosen_class_ids = [0]
        self.check_chosenClassIds(chosen_class_ids)
        self.chosen_class_ids = chosen_class_ids

    def set_chosenClassIds(self, chosen_class_ids):
        self.check_chosenClassIds(chosen_class_ids)
        self.chosen_class_ids = chosen_class_ids

    def check_chosenClassIds(self, chosen_class_ids):
        if not isinstance(chosen_class_ids, list):
            raise Exception("Chosen class ids is not a list type.")
        if max(chosen_class_ids) > max(self.frameProcessor.bboxPredictor.m_yolo_model.names.keys()):
            raise Exception(f"Invalid class id, max id is: {max(self.frameProcessor.bboxPredictor.m_yolo_model.names.keys())}")
        if min(chosen_class_ids) < min(self.frameProcessor.bboxPredictor.m_yolo_model.names.keys()):
            raise Exception(f"Invalid class id, min id is: {min(self.frameProcessor.bboxPredictor.m_yolo_model.names.keys())}")
    def set_inputVideo(self, input_video):
        if not os.path.exists(input_video):
            raise Exception("Input file does not exists.")
        pattern = r".*.[Mm][Pp]4"
        if not re.match(pattern, input_video):
            raise Exception("Input file is not mp4 format.")
        self.input_video = input_video

    def set_MTT(self, MTT):
        self.MTT = MTT

    def set_frameProcessor(self, frameProcessor):
        self.frameProcessor = frameProcessor

    def set_outputVideo(self, output_video):
        self.output_video = output_video

    def set_device(self, device):
        if device != "cpu" and not torch.cuda.is_available():
            warnings.warn("GPU is not available, setting device to cpu.")
            self.device = "cpu"
            return
        elif device != "cpu" and torch.cuda.is_available():
            self.device = device
        elif device == "cpu":
            self.device = "cpu"
        return

    def checkClassMembers(self):
        if not isinstance(self.MTT, TargetTracker):
            raise Exception("MTT is not set")
        if not isinstance(self.frameProcessor, FrameProcessing):
            raise Exception("frameProcessor is not set")
        if self.output_video == None:
            raise Exception("Output video is not set")
        if self.input_video == None:
            raise Exception("Input video is not set")

    def get_videoDimensions(self, cap):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameShape = (width, height)
        return width, height

    def get_videoFps(self, cap):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("fps: ", fps)
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

    def resize_masks(self, masks):
        if masks.ndim == 3:
            if masks[0].shape != self.frameShape:
                new_masks = np.zeros(shape=(masks.shape[0], self.frameShape[1], self.frameShape[0]), dtype=np.uint8)
                for i, mask in enumerate(masks):
                    new_masks[i] = cv2.resize(mask, self.frameShape)
                return new_masks
            return masks.astype(np.uint8)

        elif masks.ndim == 4:
            new_masks = np.zeros(shape=(masks.shape[0], self.frameShape[1], self.frameShape[0]), dtype=np.uint8)
            if masks[0][0].T.shape != self.frameShape:
                masks = masks.transpose(1, 0, 2, 3)
                for i, mask in enumerate(masks[0]):
                    new_masks[i] = cv2.resize(mask, self.frameShape)
                return new_masks
            else:
                masks = masks.transpose(1, 0, 2, 3)
                return masks[0].astype(np.uint8)
        raise Exception("resize masks not supported")


    def filterClasses(self, cls, xyxy, conf, masks):
        classes_mask = np.isin(cls, self.chosen_class_ids)
        conf = conf[classes_mask]
        xyxy = xyxy[classes_mask, :]
        masks = masks[classes_mask]
        cls = cls[classes_mask]
        return xyxy, conf, masks, cls

    def showMaskWithLabel(self, mask, frame, label, xyxy):
        color = (0,0,128) # red
        mask = self.add_color(mask, self.get_color(color)).astype(np.uint8)
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        color = (0, 0, 0)
        frame = cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def mergeMasks(self, masks, color = (0,0,128)):
        color = self.get_color(color)
        merged_masks = self.add_color(masks[0], color)
        if len(masks) == 1:
            return merged_masks.astype(np.uint8)
        for i in range(1, len(masks)):
            curr_colored_mask = self.add_color(masks[i], color)
            merged_masks = np.bitwise_or(merged_masks, curr_colored_mask)

        return merged_masks.astype(np.uint8)

    def showAllMasks(self, masks, frame, color = (0,0,128)):
        masks = self.mergeMasks(masks, color)
        frame = cv2.addWeighted(frame, 1, masks, 0.2, 0)
        return frame

    def showBboxWithLabel(self, xyxy, frame, label=""):
        color = (0, 0, 0)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.get_color(color), 2)
        frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def showAllBboxesWithLabels(self, xyxy, frame, labels=None):
        if labels is not None and len(labels) != len(xyxy):
            raise Exception("xyxy length and labels length must be the same")
        if labels is None:
            labels = [""] * len(xyxy)
        for xyxy_, label in zip(xyxy, labels):
            frame = self.showBboxWithLabel(xyxy_, frame, label)
        return frame

    # Merge masks into a single, multi-colored mask

    def show_frame(self, frame, text):
        cv2.namedWindow(f"{text}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{text}", 800, 600)
        cv2.imshow(f"{text}", frame)
        cv2.waitKey(0)

    def get_bboxCenters(self, xyxy, frame, show=True):
        bbox_centers = np.zeros(shape=(xyxy.shape[1], 2))
        for i, obj_xyxy in enumerate(xyxy):
            center_x = obj_xyxy[2] + obj_xyxy[0]
            center_y = obj_xyxy[3] + obj_xyxy[1]
            center = np.array([int(center_x / 2), int(center_y / 2)])
            bbox_centers[i] = center
            if show:
                radius = 2
                color = (0, 128, 255)
                thickness = -1
                frame = cv2.circle(frame, center, radius, color, thickness)

        return frame, bbox_centers

    def get_masksCenters(self, masks, frame, show=True):
        masks_centers = np.zeros(shape=(masks.shape[0], 2))
        for i, mask in enumerate(masks):
            y_positions, x_positions = np.nonzero(masks[i])
            center = (int(np.mean(x_positions)), int(np.mean(y_positions)))
            masks_centers[i] = np.array(center)
            if show:
                radius = 2
                color = (0, 255, 255)
                thickness = -1
                frame = cv2.circle(frame, center, radius, color, thickness)
        return frame, masks_centers

    def showLabel(self, frame, x, y, label=""):
        color = (0, 0, 0)
        frame = cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame
    def showAllLabels(self, frame, bboxes, labels):
        assert len(bboxes) == len(labels)
        for bbox, label in zip(bboxes, labels):
            x = int(bbox[0])
            y = int(bbox[1]) - 10
            frame = self.showLabel(frame, x, y, str(round(label,5)))
        return frame

    def run(self, P):
        self.checkClassMembers()
        # videoCap = cv2.VideoCapture(self.m_input_video)
        videoCap = Stream(self.input_video, 10)
        output_video_boxes = self.get_outputVideoWriter(videoCap, self.output_video + "_boxes.mp4")
        output_video_masks = self.get_outputVideoWriter(videoCap, self.output_video + "_masks.mp4")
        frame_num = 1

        width, height = self.get_videoDimensions(videoCap)
        self.frameProcessor.set_frameDimensions((width, height))
        self.MTT.set_ImageSize(np.array([width, height]))
        self.MTT.add_PerimeterSpawnPoints(w=0.1, cov=P)

        while videoCap.isOpened():
            print("frame: ", frame_num)
            print("================================")
            ret, frame = videoCap.read()
            if not ret:
                break
            if frame is None:
                continue
            bboxes, masks = self.frameProcessor.predict(frame)

            frame_copy = frame.copy()
            frameWithSpawnPoints = self.MTT.show_SpawnPoints(frame_copy)
            if len(bboxes) == 0 or masks is None or masks.shape[0] == 0:
                self.show_frame(frameWithSpawnPoints,frame_num)
                frame_num +=1
                continue
            xyxy, conf, masks, cls = self.filterClasses(bboxes.cls.numpy(), bboxes.xyxy.numpy(), bboxes.conf.numpy(), masks)
            if len(xyxy) == 0 or masks is None or masks.shape[0] == 0:
                self.show_frame(frameWithSpawnPoints,frame_num)
                frame_num +=1
                continue

            masks = self.resize_masks(masks)
            frameWithBboxes, z_bboxes_centers = self.get_bboxCenters(xyxy, frameWithSpawnPoints, True)
            frameWithBboxes, z_masks_centers = self.get_masksCenters(masks, frameWithBboxes, True)
            frameWithBboxes = self.showAllBboxesWithLabels(xyxy, frameWithBboxes)


            self.MTT.predict(frame_num)
            self.MTT.update(z_masks_centers, conf, xyxy, masks, frame, frame_num)
            # self.MTT.pruneByMaxWeight(0.05)
            if frame_num < 20 or 1:
                self.MTT.mergeTargets()
            print("Trackers: ", len(self.MTT.trackers))

            predicted_xyxy = []
            predicted_pd = []
            predicted_cls = []

            prev_masks = []
            act_masks = []
            for i, target in enumerate(self.MTT.trackers):
                center = (int(target.m[0]), int(target.m[1]))
                frameWithBboxes = cv2_confidence_ellipse(center=center, cov_matrix=target.P, image=frameWithBboxes, showCenter=True)
                # print("target xyxy: ", target.xyxy)
                if target.xyxy is not None:
                    predicted_xyxy.append(target.xyxy)
                    predicted_pd.append(target.pd)
                    predicted_cls.append(1)

                    print(f"target {i}:")
                if target.mask is not None:
                    print(f"    mask mean: ", np.mean(frame[target.mask.nonzero()]))
                    act_masks.append(target.mask)
                if target.prev_xyxy is not None:
                    m = np.mean(frame[int(target.prev_xyxy[1]):int(target.prev_xyxy[3]),
                                    int(target.prev_xyxy[0]):int(target.prev_xyxy[2]), 0])
                    print("     prev bbox mean: ", m)
                if target.objectStats is not None:
                    prev_masks.append(target.objectStats.mask)
                print("     w: ", target.w)
                    # for prev in prev_masks:
                    #     print("prev mask: ")
                    #     msk = prev * 255
                    #     cv2.imshow(f"prev_{frame_num}", msk)
                    #     cv2.waitKey(0)



                    # cv2.imshow(f"prev_{frame_num}", frameWithBboxes)
                    # cv2.waitKey(0)
                    # m = np.mean(frame[int(target.xyxy[1]):int(target.xyxy[3]),
                    #             int(target.xyxy[0]):int(target.xyxy[2]), 0])
                    # print("     bbox mean: ", m)

            frameWithBboxes = self.showAllLabels(frameWithBboxes, predicted_xyxy, predicted_pd)
            cls = np.zeros(shape=len(prev_masks))
            print("prev mask len: ", len(prev_masks))
            if len(prev_masks) > 0:
                frameWithBboxes = self.showAllMasks(prev_masks, frameWithBboxes, (128, 0, 0))
                # merged_colored_mask = self.merge_masks_colored(prev_masks, cls)
                # frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)
            cls = np.zeros(shape=len(act_masks))
            if len(act_masks) > 0:
                frameWithBboxes = self.showAllMasks(act_masks, frameWithBboxes, color=(0, 0, 128))
                # merged_colored_mask = self.merge_masks_colored(act_masks, cls)
                # frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)

            # frameWithBboxes = self.showAllMasks(masks, frameWithBboxes, (0, 255, 255))

            # print("xyxy: ", xyxy)
            # print("predicted xyxy: ", predicted_xyxy)
            # frameWithBboxes = self.frameProcessor.visualizeDetectionsBbox(frameWithBboxes,
            #                                                               predicted_xyxy,
            #                                                               predicted_conf,
            #                                                               predicted_cls)
            act_masks = np.array(act_masks, dtype=np.uint8)
            print("act mask ddtpye: ", act_masks.dtype)
            print("frame dtpye: ", frameWithBboxes.dtype)
            # if len(prev_masks) > 0 and len(act_masks) > 0:
            #     for xyxy, mask in zip(xyxy, act_masks):
            #         frameWithBboxes = self.showMaskWithLabel(mask, frameWithBboxes, "tmp", xyxy.astype(int))
            output_video_boxes.write(frameWithBboxes)

            # merged_colored_mask = self.merge_masks_colored(masks, bboxes.cls)
            # print("merge mask shape: ", merged_colored_mask.shape)
            # print("merge mask dtype: ", merged_colored_mask.dtype)
            # print("frame with boxes shape: ", frameWithBboxes.shape)
            # print("frame with boxes dtype: ", frameWithBboxes.dtype)
            # frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)
            # if masks is not None:
            #     masks = self.resize_masks(masks)
            #     for mask in masks:
            #         y, x = np.indices(mask.shape)
            #         print(type(mask))
            #         print(mask.shape)
            #         print(mask)
            #         print("max: ", np.max(mask))
            #         print("min: ", np.min(mask))
            #         mask = np.array(mask, dtype=np.bool_)
            #         # Use the mask to filter x and y coordinates
            #         x_positions = x[mask]
            #         y_positions = y[mask]
            #
            #         # Calculate the mean x and y positions
            #         mean_x = int(np.mean(x_positions))
            #         mean_y =int( np.mean(y_positions))
            #         print("mean x: ", mean_x)
            #         print("mean y: ", mean_y)
            #         center = (mean_x, mean_y)
            #         color = (0, 255, 128)
            #         radius = 2
            #         thickness = -1
            #         frameWithBboxes = cv2.circle(frameWithBboxes, center, radius, color, thickness)
            #
            #     merged_colored_mask = self.merge_masks_colored(masks, bboxes.cls)
            #     frameWithBboxes = cv2.addWeighted(frameWithBboxes, 0.7, merged_colored_mask, 0.7, 0)


            cv2.namedWindow(f"{frame_num}", cv2.WINDOW_NORMAL)
            # Using resizeWindow()
            cv2.resizeWindow(f"{frame_num}", 800, 600)

            cv2.imshow(f"{frame_num}", frameWithBboxes)
            cv2.waitKey(0)

            # frameWithYoloDetections = Image.fromarray(frameWithYoloDetections, "RGB")
            #
            # frameWithYoloDetections.show(title=f"num_{frame_num}")
            # output_video_boxes.write(frame)
            """

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
            print("mask shape: ", masks[0].shape)
            print("frame[:,:,0] ", frame[:,:,0].shape)
            print("masks[0][0] ", masks[0][0].shape)
            frame_mask = np.ma.array(frame[:,:,0], mask = np.invert(masks[0][0]))
            print("frame * mask mean", frame_mask.mean())
            merged_colored_mask = self.merge_masks_colored(masks, yoloDetections[0].boxes.cls)

            # Write masks to output video
            image_combined = cv2.addWeighted(frame, 0.7, merged_colored_mask, 0.7, 0)
            output_video_masks.write(image_combined)
            """
            frame_num += 1
            if frame_num > 60 and 0:
                break
        videoCap.release()
        output_video_boxes.release()
        output_video_masks.release()
        cv2.destroyAllWindows()
