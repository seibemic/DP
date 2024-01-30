from ultralytics import YOLO
import os
import numpy as np
import cv2

class YOLOHandler:
    def __init__(self, chosen_class_ids=[0]):
        HOME = os.getcwd()
        self.m_yolo_model = YOLO(f'{HOME}/weights/yolov8n.pt')
        if not isinstance(chosen_class_ids, list):
            raise Exception("Chosen class ids is not a list type.")
        if max(chosen_class_ids) > max(self.m_yolo_model.names.keys()):
            raise Exception(f"Invalid class id, max id is: {max(self.m_yolo_model.names.keys())}")
        if min(chosen_class_ids) < min(self.m_yolo_model.names.keys()):
            raise Exception(f"Invalid class id, min id is: {min(self.m_yolo_model.names.keys())}")
        self.chosen_class_ids = chosen_class_ids
        self.colors = np.random.randint(0, 256, size=(len(self.m_yolo_model.names), 3))

    def PrintAvailableModelNames(self):
        print(self.m_yolo_model.names)

    def set_chosenClassIds(self, chosen_class_ids):
        self.chosen_class_ids = chosen_class_ids

    def predict(self, frame):
        detections = self.m_yolo_model.predict(frame, conf=0.7)
        return detections

    def get_color(self, color):
        return int(color[0]), int(color[1]), int(color[2])

    def visualizeDetectionsBbox(self, frame, boxes, conf_thresholds, class_ids):
        frame_copy = np.copy(frame)
        for idx in range(len(boxes)):
            class_id = int(class_ids[idx])
            conf = float(conf_thresholds[idx])
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            color = self.colors[class_id]
            label = f"{self.m_yolo_model.names[class_id]}: {conf:.2f}"
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), self.get_color(color), 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.get_color(color), 2)
        return frame_copy