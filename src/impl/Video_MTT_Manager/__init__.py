import torch
import warnings
import cv2
import os
import re
import numpy as np

from src.impl.Frame_Processing import FrameProcessing
from src.impl.MTT import TargetTracker
from src.impl.MTT.confidence_ellipse import cv2_confidence_ellipse

class VideoMTT:
    def __init__(self, input_video=None, MTT=None, frameProcessor=None, output_video=None, chosen_class_ids=None):
        self.frameShape = None
        self.m_output_video = None
        self.frameProcessor = frameProcessor
        self.m_MTT = None
        self.m_input_video = None
        self.set_inputVideo(input_video)
        self.set_MTT(MTT)
        self.set_outputVideo(output_video)
        self.m_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
        self.m_input_video = input_video

    def set_MTT(self, MTT):
        self.m_MTT = MTT

    def set_frameProcessor(self, frameProcessor):
        self.frameProcessor = frameProcessor

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
        if not isinstance(self.m_MTT, TargetTracker):
            raise Exception("MTT is not set")
        if not isinstance(self.frameProcessor, FrameProcessing):
            raise Exception("frameProcessor is not set")
        if self.m_output_video == None:
            raise Exception("Output video is not set")
        if self.m_input_video == None:
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
        print(masks.ndim)
        if masks.ndim == 3:
            if masks[0].shape != self.frameShape:
                new_masks = np.zeros(shape=(masks.shape[0], self.frameShape[1], self.frameShape[0]), dtype=np.int8)
                for i, mask in enumerate(masks):
                    new_masks[i] = cv2.resize(mask, self.frameShape)
                return new_masks
            return masks.astype(np.int8)

        elif masks.ndim == 4:
            new_masks = np.zeros(shape=(masks.shape[0], self.frameShape[1], self.frameShape[0]), dtype=np.int8)
            if masks[0][0].T.shape != self.frameShape:
                masks = masks.transpose(1, 0, 2, 3)
                for i, mask in enumerate(masks[0]):
                    new_masks[i] = cv2.resize(mask, self.frameShape)
                return new_masks
            else:
                masks = masks.transpose(1, 0, 2, 3)
                return masks[0].astype(np.int8)
        raise Exception("resize masks not supported")


    def filterClasses(self, cls, xyxy, conf, masks):
        classes_mask = np.isin(cls, self.chosen_class_ids)
        conf = conf[classes_mask]
        xyxy = xyxy[classes_mask, :]
        masks = masks[classes_mask]
        return xyxy, conf, masks


    # Merge masks into a single, multi-colored mask
    def merge_masks_colored(self, masks, class_ids):
        filtered_class_ids = []
        filtered_masks = []
        for idx, cid in enumerate(class_ids):
            if int(cid) in self.chosen_class_ids:
                filtered_class_ids.append(cid)
                filtered_masks.append(masks[idx])
        merged_with_colors = self.add_color(filtered_masks[0],
                                            self.get_color(self.frameProcessor.colors[int(filtered_class_ids[0])])).astype(
            np.uint8)
        if len(filtered_masks) == 1:
            return merged_with_colors

        for i in range(1, len(filtered_masks)):
            curr_mask_with_colors = self.add_color(filtered_masks[i], self.get_color(
                self.frameProcessor.colors[int(filtered_class_ids[i])]))
            merged_with_colors = np.bitwise_or(merged_with_colors, curr_mask_with_colors)
        return merged_with_colors.astype(np.uint8)

    def show_frame(self, frame, text):
        cv2.namedWindow(f"{text}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{text}", 1600, 1200)
        cv2.imshow(f"{text}", frame)
        cv2.waitKey(0)

    def run(self, P):
        self.checkClassMembers()
        videoCap = cv2.VideoCapture(self.m_input_video)
        output_video_boxes = self.get_outputVideoWriter(videoCap, self.m_output_video + "_boxes.mp4")
        output_video_masks = self.get_outputVideoWriter(videoCap, self.m_output_video + "_masks.mp4")
        frame_num = 1

        width, height = self.get_videoDimensions(videoCap)
        self.frameProcessor.set_frameDimensions((width, height))
        self.m_MTT.set_ImageSize(np.array([width, height]))
        self.m_MTT.add_PerimeterSpawnPoints(w=0.5, cov=P)

        while videoCap.isOpened():
            print("frame: ", frame_num)
            print("================================")
            ret, frame = videoCap.read()
            if not ret:
                break
            bboxes, masks = self.frameProcessor.predict(frame)
            frameWithSpawnPoints = self.m_MTT.show_SpawnPoints(frame)
            if len(bboxes) == 0:
                self.show_frame(frameWithSpawnPoints,frame_num)
                frame_num +=1
                continue

            frameWithBboxes = self.frameProcessor.visualizeDetectionsBbox(frameWithSpawnPoints,
                                                                          bboxes.xyxy,
                                                                          bboxes.conf,
                                                                          bboxes.cls)
            z_bboxes_center = []
            z_masks_center = []

            xyxy, conf, masks = self.filterClasses(bboxes.cls.numpy(), bboxes.xyxy.numpy(), bboxes.conf.numpy(), masks)
            masks = self.resize_masks(masks)
            for i, obj_xyxy in enumerate(xyxy):
                center_x = obj_xyxy[2] + obj_xyxy[0]
                center_y = obj_xyxy[3] + obj_xyxy[1]
                center = np.array([int(center_x / 2), int(center_y / 2)])
                # m = np.mean(frame[int(obj_xyxy[1]):int(obj_xyxy[3]),
                #                      int(obj_xyxy[0]):int(obj_xyxy[2]),0])
                # boxes.append(m)
                z_bboxes_center.append(center)
                radius = 2
                color = (0, 128, 255)
                thickness = -1
                frameWithBboxes = cv2.circle(frameWithBboxes, center, radius, color, thickness)
                y_positions, x_positions = np.nonzero(masks[i])
                mean_x = int(np.mean(x_positions))
                mean_y = int(np.mean(y_positions))
                center = (mean_x, mean_y)
                color = (0, 255, 128)
                radius = 2
                thickness = -1
                frameWithBboxes = cv2.circle(frameWithBboxes, center, radius, color, thickness)
                z_masks_center.append(center)



            self.m_MTT.predict()

            self.m_MTT.update(np.array(z_masks_center), conf, xyxy, masks, frame)
            self.m_MTT.pruneByMaxWeight(0.1)
            # self.m_MTT.mergeTargets()
            print("Trackers: ", len(self.m_MTT.trackers))

            predicted_xyxy = []
            predicted_conf = []
            predicted_cls = []

            prev_masks = []
            act_masks = []
            for i, target in enumerate(self.m_MTT.trackers):
                radius = 2
                color = (0, 0, 255)
                thickness = -1
                center_x = int(target.m[0])
                center_y = int(
                    target.m[1])
                center = (center_x, center_y)
                print("center: ", center, "w: ", target.w)
                frameWithBboxes = cv2.circle(frameWithBboxes, center, radius, color, thickness)
                frameWithBboxes = cv2_confidence_ellipse(center, target.P, frameWithBboxes, color=color,
                                                                 thickness=1)
                # print("target xyxy: ", target.xyxy)
                if target.xyxy is not None:
                    predicted_xyxy.append(target.xyxy)
                    predicted_conf.append(target.conf)
                    predicted_cls.append(1)

                    print(f"target {i}:")
                    if target.mask is not None:
                        print(f"    mask mean: ", np.mean(frame[target.mask.nonzero()]))
                        act_masks.append(target.mask)
                    if target.prev_xyxy is not None:
                        m = np.mean(frame[int(target.prev_xyxy[1]):int(target.prev_xyxy[3]),
                                    int(target.prev_xyxy[0]):int(target.prev_xyxy[2]), 0])
                        print("     prev bbox mean: ", m)
                    if target.prev_mask is not None:
                        prev_masks.append(target.prev_mask)
                    # for prev in prev_masks:
                    #     print("prev mask: ")
                    #     msk = prev * 255
                    #     cv2.imshow(f"prev_{frame_num}", msk)
                    #     cv2.waitKey(0)

                    print("prev mask len: ", len(prev_masks))

                    # cv2.imshow(f"prev_{frame_num}", frameWithBboxes)
                    # cv2.waitKey(0)
                    m = np.mean(frame[int(target.xyxy[1]):int(target.xyxy[3]),
                                int(target.xyxy[0]):int(target.xyxy[2]), 0])
                    print("     bbox mean: ", m)
            cls = np.zeros(shape=len(prev_masks))
            if len(prev_masks) > 0:
                merged_colored_mask = self.merge_masks_colored(prev_masks, cls)
                frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)
            cls = np.zeros(shape=len(act_masks))
            if len(act_masks) > 0:
                merged_colored_mask = self.merge_masks_colored(act_masks, cls)
                frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)
            # print("xyxy: ", xyxy)
            # print("predicted xyxy: ", predicted_xyxy)
            # frameWithBboxes = self.frameProcessor.visualizeDetectionsBbox(frameWithBboxes,
            #                                                               predicted_xyxy,
            #                                                               predicted_conf,
            #                                                               predicted_cls)

            output_video_boxes.write(frameWithBboxes)

            # merged_colored_mask = self.merge_masks_colored(masks, bboxes.cls)
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
            cv2.resizeWindow(f"{frame_num}", 1600, 1200)

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
    def run2(self):
        self.checkClassMembers()
        videoCap = cv2.VideoCapture(self.m_input_video)
        output_video_boxes = self.get_outputVideoWriter(videoCap, self.m_output_video + "_boxes.mp4")
        output_video_masks = self.get_outputVideoWriter(videoCap, self.m_output_video + "_masks.mp4")
        # print(self.m_input_video)
        frame_num = 1

        d = 400
        P = np.array([[d, 0, 0, 0],
                      [0, d, 0, 0],
                      [0, 0, d, 0],
                      [0, 0, 0, d]])
        width, height = self.get_videoDimensions(videoCap)
        self.m_MTT.set_ImageSize(np.array([width, height]))
        self.m_MTT.add_PerimeterSpawnPoints(w=0.5, cov=P)
        while videoCap.isOpened():
            print("frame: ", frame_num)
            print("================================")
            ret, frame = videoCap.read()
            if not ret:
                break
            yoloDetections = self.m_YOLO.predict(frame)
            frameWithYoloDetections = self.m_YOLO.visualizeDetectionsBbox(frame,
                                                                          yoloDetections[0].boxes.cpu().xyxy,
                                                                          yoloDetections[0].boxes.cpu().conf,
                                                                          yoloDetections[0].boxes.cpu().cls)
            # for mask in yoloDetections[0].masks.data:
            #     mask = mask.numpy()*255
            #     mask = cv2.resize(mask,(width,height))
            #     cv2.imshow(f"{frame_num}", mask)
            #     cv2.waitKey(0)
            # frameWithYoloDetections = self.merge_masks_colored(yoloDetections[0].masks.data.numpy()*255, yoloDetections[0].boxes.cls)
            # print("masK: ")
            # print(yoloDetections[0].masks.data.numpy())
            z = []
            boxes = []
            print("shape: ", frame.shape)
            for obj_xyxy in yoloDetections[0].boxes.cpu().xyxy:
                center_x = obj_xyxy[2].numpy() + obj_xyxy[0].numpy()
                center_y = obj_xyxy[3].numpy() + obj_xyxy[1].numpy()
                center = np.array([int(center_x / 2), int(center_y / 2)])
                print("x,y,x,y: ", obj_xyxy[0].numpy(), obj_xyxy[1].numpy(),obj_xyxy[2].numpy(),obj_xyxy[3].numpy())
                # print("frame: ",frame[int(obj_xyxy[0].numpy()):int(obj_xyxy[2].numpy()),
                #                      int(obj_xyxy[1].numpy()):int(obj_xyxy[3].numpy()),0])
                m = np.mean(frame[int(obj_xyxy[1].numpy()):int(obj_xyxy[3].numpy()),
                                     int(obj_xyxy[0].numpy()):int(obj_xyxy[2].numpy()),0])
                radius = 2
                color = (0, 128, 255)
                thickness = -1
                frameWithYoloDetections = cv2.circle(frameWithYoloDetections, center, radius, color, thickness)

                boxes.append(m)
                print("mean: ", m)
                z.append(center)

            self.m_MTT.predict()

            self.m_MTT.update(np.array(z))
            self.m_MTT.pruneByMaxWeight(0.1)
            self.m_MTT.mergeTargets()
            print("Trackers: ", len(self.m_MTT.trackers))
            # print(z)
            for target in self.m_MTT.trackers:
                radius = 2
                color = (0, 0, 255)
                thickness = -1
                center_x = int(target.m[0])
                center_y = int(target.m[1])
                center = (center_x, center_y)
                print("center: ", center, "w: ", target.w)
                frameWithYoloDetections = cv2.circle(frameWithYoloDetections, center, radius, color, thickness)
                frameWithYoloDetections = cv2_confidence_ellipse(center, target.P, frameWithYoloDetections, color=color,
                                                                 thickness=1)
            frameWithYoloDetections = self.m_MTT.show_SpawnPoints(frameWithYoloDetections)
            output_video_boxes.write(frameWithYoloDetections)

            cv2.namedWindow(f"{frame_num}", cv2.WINDOW_NORMAL)
            # Using resizeWindow()
            cv2.resizeWindow(f"{frame_num}", 1600, 1200)

            cv2.imshow(f"{frame_num}", frameWithYoloDetections)
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
