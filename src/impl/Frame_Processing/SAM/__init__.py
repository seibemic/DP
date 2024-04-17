import cv2
import PIL
from PIL import Image
import numpy as np
import torch
import os
import warnings

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class SAM_handler:
    def __init__(self, device):
        HOME = os.getcwd()
        if not os.path.exists(f"{HOME}/weights/sam_vit_h_4b8939.pth"):
            warnings.warn("sam not available, downloading...")
            os.system(f"""wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth""")
            os.system(f"""mv sam_vit_h_4b8939.pth {HOME}/weights/sam_vit_h_4b8939.pth""")
        self.m_sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
        self.m_model_type = "vit_h"
        self.m_device = device
        self.sam = sam_model_registry[self.m_model_type](checkpoint=self.m_sam_checkpoint)
        self.sam.to(device=self.m_device)
        self.m_predictor = SamPredictor(self.sam)

    def transformBoxes(self, frameDimensions, detections):
        transformed_boxes = self.m_predictor.transform.apply_boxes_torch(detections.xyxy,
                                                                         frameDimensions)
        return transformed_boxes

    def predict(self, frame, frameDimensions, detections):
        transformed_boxes = self.transformBoxes(frameDimensions, detections)
        self.m_predictor.set_image(frame)
        masks, scores, logits = self.m_predictor.predict_torch(
            boxes=transformed_boxes,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )
        masks = np.array(masks.cpu())
        return masks, scores, logits