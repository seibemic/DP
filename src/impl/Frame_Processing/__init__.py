import torch
from src.impl.Frame_Processing.YOLO import YOLOHandler
from src.impl.Frame_Processing.SAM import SAM_handler


class FrameProcessing:
    def __init__(self, mode=0, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mode = mode

        self.bboxes = None
        self.masks = None
        self.scores = None
        self.logits = None
        self.colors = None

        self.bboxPredictor = None
        self.maskPredictor = None
        self.set_Predictors()

    def set_Predictors(self):
        if self.mode == 0:  # yolo, yolo
            self.bboxPredictor = YOLOHandler(segment=True)
            self.colors = self.bboxPredictor.colors
        elif self.mode == 1:  # yolo, sam
            self.bboxPredictor = YOLOHandler(segment=False)
            self.maskPredictor = SAM_handler(self.device)
            self.colors = self.bboxPredictor.colors

    def predict(self, frame):
        if self.mode == 0:  # yolo, yolo
            detections = self.bboxPredictor.predict(frame)
            self.bboxes = detections[0].boxes.cpu()
            if detections[0].masks:
                self.masks = detections[0].masks.data.numpy()
            else:
                self.masks = None
        elif self.mode == 1:  # yolo, sam
            detections = self.bboxPredictor.predict(frame)
            self.bboxes = detections[0].boxes.cpu()
            self.masks, self.scores, self.logits = self.maskPredictor.predict(frame)
        return self.bboxes, self.masks

    def get_bboxes(self):
        return self.bboxes

    def get_masks(self):
        return self.masks

    def get_scores(self):
        return self.scores

    def get_logits(self):
        return self.logits

    def printAvailableModelNames(self):
        self.bboxPredictor.PrintAvailableModelNames()

    def visualizeDetectionsBbox(self, frame, boxes, conf_thresholds, class_ids):
        return self.bboxPredictor.visualizeDetectionsBbox(frame, boxes, conf_thresholds, class_ids)
