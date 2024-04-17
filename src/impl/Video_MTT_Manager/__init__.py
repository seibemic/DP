import torch
import warnings
import cv2
import os
import re
import numpy as np

from src.impl.Frame_Processing import FrameProcessing, DINO_handler
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
        if not isinstance(self.frameProcessor.bboxPredictor, DINO_handler):
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
        if masks is None or masks.shape[0] == 0:
            return None
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
        if masks is None:
            return xyxy, conf, masks, cls
        classes_mask = np.isin(cls, self.chosen_class_ids)
        conf = conf[classes_mask]
        xyxy = xyxy[classes_mask, :]
        masks = masks[classes_mask]
        cls = cls[classes_mask]
        return xyxy, conf, masks, cls

    def showMaskWithLabel(self, mask, frame, label, xyxy):
        color = (0,0,128) # red
        mask = self.add_color(mask, self.get_color(color)).astype(np.uint8)
        frame = cv2.addWeighted(frame, 0.7, mask, 0.8, 0)
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
        frame = cv2.addWeighted(frame, 1, masks, 0.8, 0)#0.3
        return frame

    def showBboxWithLabel(self, xyxy, frame, label="", color=(0, 0, 0)):
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        # print("xyxy: ", xyxy)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.get_color(color), 2)
        frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def showAllBboxesWithLabels(self, xyxy, frame, labels=None, color = (0, 0, 0)):
        if labels is not None and len(labels) != len(xyxy):
            raise Exception("xyxy length and labels length must be the same")
        if labels is None:
            labels = [""] * len(xyxy)
        for xyxy_, label in zip(xyxy, labels):
            frame = self.showBboxWithLabel(xyxy_, frame, label, color)
        return frame

    # Merge masks into a single, multi-colored mask

    def show_frame(self, frame, text):
        cv2.namedWindow(f"{text}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{text}", 800, 600)
        cv2.imshow(f"{text}", frame)
        cv2.waitKey(0)

    def get_bboxCenters(self, xyxy, frame, show=True):
        bbox_centers = np.zeros(shape=(xyxy.shape[0], 2))
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
        if masks is None:
            masks_centers= []
            return frame, np.array(masks_centers)
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
        videoCap = Stream(self.input_video, 29)
        output_video_boxes = self.get_outputVideoWriter(videoCap, self.output_video + "_boxes.mp4")
        output_video_masks = self.get_outputVideoWriter(videoCap, self.output_video + "_masks.mp4")
        frame_num = 1

        width, height = self.get_videoDimensions(videoCap)
        self.frameProcessor.set_frameDimensions((width, height))
        self.MTT.set_ImageSize(np.array([width, height]))
        #self.MTT.add_PerimeterSpawnPoints(w=0.1, cov=P)
        # self.MTT.add_MeshSpawnPoints(w=0.1,cov=P)
        # self.MTT.add_SpawnPoint(np.array([708,1254]), w=0.1, cov=P)
       # self.MTT.add_SpawnPoint(np.array([617, 1050]), w=0.1, cov=P)
        # self.MTT.add_SpawnPoint(np.array([604, 800]), w=0.1, cov=P)
        # self.MTT.add_SpawnPoint(np.array([460, 542]), w=0.1, cov=P)
        # self.MTT.add_SpawnPoint(np.array([281, 542]), w=0.1, cov=P)
        self.MTT.add_SpawnPoint(np.array([1480, 980]), w=0.1, cov=P)
        self.MTT.add_SpawnPoint(np.array([1726, 913]), w=0.1, cov=P)
        self.MTT.add_SpawnPoint(np.array([1752, 771]), w=0.1, cov=P)
        self.MTT.add_SpawnPoint(np.array([1822, 708]), w=0.1, cov=P)

        #self.MTT.add_SpawnPoint(np.array([790, 420]), w=0.1, cov=P/3)
        #self.MTT.add_SpawnPoint(np.array([872, 420]), w=0.1, cov=P/3)
        road = cv2.imread("/home/michal/Documents/FIT/DP/dp/src/data/imgs/road2.png", cv2.IMREAD_UNCHANGED)
        print("road shape:")
        print(road.shape)
        desired_width = 1120
        desired_height = 370

        # Resize the image
        road = cv2.resize(road, (desired_width, desired_height))
        rotation_angle = 0
        rows, cols, _ = road.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        road = cv2.warpAffine(road, M, (cols, rows))


        mask_road = road[:, :, 3] / 255.0
        alpha_mask_inv = 1.0 - mask_road
        height1, width1, _ = road.shape
        print(road.shape)
        x_offset = 0  # X coordinate where you want to place image1 in image2
        y_offset = 350  # Y coordinate where you want to place image1 in image2
        roi = road[y_offset:y_offset + height1, x_offset:x_offset + width1]


        while videoCap.isOpened():
            print("frame: ", frame_num)
            print("================================")
            ret, frame = videoCap.read()
            if not ret:
                break
            if frame is None:
                continue

            # frame[300:800,485:740,:] = 255
           # frame = frame[665:1200, 400:900]
           # frame[y_offset:y_offset+height1, x_offset:x_offset+width1] = road


            # frame[y_offset:y_offset + height1, x_offset:x_offset + width1] = \
            #     frame[y_offset:y_offset + height1, x_offset:x_offset + width1] * (1 - mask_road) + \
            #     road[:, :, :3] * mask_road
            addRoad = 0
            if addRoad:
                for c in range(0, 3):
                    frame[y_offset:y_offset + height1, x_offset:x_offset + width1, c] = \
                        (mask_road * road[:, :, c] +
                         alpha_mask_inv * frame[y_offset:y_offset + height1, x_offset:x_offset + width1, c])
            frame_obstacles = frame.copy()
            addObstacle = 1
            if addObstacle:
                height, width = frame_obstacles.shape[:2]
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                pts_left = np.array([[0, 0], [0, 600], [1400, 0]], np.int32)
                cv2.fillPoly(canvas, [pts_left], (255, 255, 255))
                pts_right = np.array([[width, 0], [width, 600], [width - 1400, 0]], np.int32)
                cv2.fillPoly(canvas, [pts_right], (255, 255, 255))
                rect_left = (0, 0,980, 1120)
                cv2.rectangle(canvas, (rect_left[0], rect_left[1]),
                              (rect_left[0] + rect_left[2], rect_left[1] + rect_left[3]), (255, 255, 255), -1)


                #frame = cv2.add(frame, canvas)
                frame_obstacles = cv2.add(frame_obstacles, canvas)
            bboxes, masks = self.frameProcessor.predict(frame_obstacles)

            frame_copy = frame.copy()
            frameWithSpawnPoints = self.MTT.show_SpawnPoints(frame_copy)
            try:
                xyxy, conf, masks, cls = self.filterClasses(bboxes.cls.numpy(), bboxes.xyxy.numpy(), bboxes.conf.numpy(), masks)
            except Exception as e:
                xyxy = bboxes.xyxy
                masks = masks
            masks = self.resize_masks(masks)

            frameWithBboxes, z_bboxes_centers = self.get_bboxCenters(xyxy, frameWithSpawnPoints, True)
            frameWithBboxes, z_masks_centers = self.get_masksCenters(masks, frameWithBboxes, True)
            frameWithBboxes = self.showAllBboxesWithLabels(xyxy+3, frameWithBboxes, color=(0,0,255))

            self.MTT.predict()
            self.MTT.update(z_masks_centers, xyxy, masks, frame, frame_num)
            self.MTT.pruneByMaxWeight(0.1)
            self.MTT.mergeTargets()
            print("Trackers: ", len(self.MTT.trackers))

            prev_xyxy = []
            predicted_xyxy = []
            predicted_pd = []
            states = []
            predicted_cls = []

            prev_masks = []
            act_masks = []
            for i, target in enumerate(self.MTT.trackers):
                center = (int(target.m[0]), int(target.m[1]))
                if target.w > 0.1:
                    frameWithBboxes = cv2_confidence_ellipse(center=center, cov_matrix=target.P, image=frameWithBboxes, showCenter=True)
                if target.objectStats is not None:
                    prev_xyxy.append(target.objectStats.xyxy)
                if target.xyxy is not None:
                    predicted_xyxy.append(target.xyxy)
                    predicted_pd.append(target.pd)
                    states.append((target.state))
                    predicted_cls.append(1)

                    print(f"target {i}:")
                if target.mask is not None:
                    print(f"    mask mean: ", np.mean(frame[target.mask.nonzero()]))
                    act_masks.append(target.mask)

                if target.objectStats is not None:
                    prev_masks.append(target.objectStats.mask)
                print("     w: ", target.w)
                print("     state: ", target.markovChain.get_probs())
                print("     m: ", target.m)
                print("     pd: ", target.pd)
                print("     P: ", np.diag(target.P))

            frameWithBboxes = self.showAllLabels(frameWithBboxes, predicted_xyxy, states)
            show = 0
            if show:
                if len(prev_xyxy) > 0:
                    frameWithBboxes = self.showAllBboxesWithLabels(prev_xyxy,frameWithBboxes,None,(255,0,0))
                if len(predicted_xyxy) > 0:
                    frameWithBboxes = self.showAllBboxesWithLabels(predicted_xyxy, frameWithBboxes, None, (0,0,0))
                    # frameWithBboxes = self.showAllLabels(frameWithBboxes, predicted_xyxy, predicted_pd)
                    frameWithBboxes = self.showAllLabels(frameWithBboxes, predicted_xyxy, states)
                cls = np.zeros(shape=len(prev_masks))
                print("prev mask len: ", len(prev_masks))
                if len(prev_masks) > 0:
                    frameWithBboxes = self.showAllMasks(prev_masks, frameWithBboxes, (255, 0, 0))
                    # merged_colored_mask = self.merge_masks_colored(prev_masks, cls)
                    # frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)
                cls = np.zeros(shape=len(act_masks))
                if len(act_masks) > 0:
                    frameWithBboxes = self.showAllMasks(act_masks, frameWithBboxes, color=(0, 0, 255))
                # merged_colored_mask = self.merge_masks_colored(act_masks, cls)
                # frameWithBboxes = cv2.addWeighted(frameWithBboxes, 1, merged_colored_mask, 0.7, 0)

            output_video_boxes.write(frameWithBboxes)


            if frame_num > 0:
                cv2.namedWindow(f"{frame_num}", cv2.WINDOW_NORMAL)
                # Using resizeWindow()

                cv2.resizeWindow(f"{frame_num}", 1200, 800)
                # frameWithBboxes=frameWithBboxes[665:1400, 400:900]
                cv2.imshow(f"{frame_num}", frameWithBboxes)
                cv2.waitKey(0)

            frame_num += 1
            if frame_num > 60 and 0:
                break
        videoCap.release()
        output_video_boxes.release()
        output_video_masks.release()
        cv2.destroyAllWindows()
