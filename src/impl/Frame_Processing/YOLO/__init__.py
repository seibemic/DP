from ultralytics import YOLO
import os
import numpy as np
import cv2

class YOLOHandler:
    def __init__(self, segment = True):
        HOME = os.getcwd()
        if segment:
            self.m_yolo_model = YOLO(f'{HOME}/weights/yolov8l-seg.pt')
        else:
            self.m_yolo_model = YOLO(f'{HOME}/weights/yolov8l.pt')
        self.segment = segment

        self.colors = np.random.randint(0, 256, size=(len(self.m_yolo_model.names), 3))

    def PrintAvailableModelNames(self):
        print(self.m_yolo_model.names)

    def predict(self, frame):
        if self.segment:
            detections = self.m_yolo_model(frame, conf = 0.3, verbose = False)
            return detections

        detections = self.m_yolo_model.predict(frame, conf=0.3, verbose=False)

        # print("detections:")
        # print(detections)
        return detections

    def get_color(self, color):
        return int(color[0]), int(color[1]), int(color[2])

    def visualizeDetectionsBbox(self, frame, boxes, conf_thresholds, class_ids):
        #frame_copy = np.copy(frame)
        for idx in range(len(boxes)):
            class_id = int(class_ids[idx])
            conf = float(conf_thresholds[idx])
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            color = self.colors[class_id]
            label = f"{self.m_yolo_model.names[class_id]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.get_color(color), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.get_color(color), 2)
        return frame