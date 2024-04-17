import numpy as np
import torch
import os
import warnings
import sys
from typing import List
class DINO_handler:
    def __init__(self, device):
        HOME = os.getcwd()
        PATH = HOME + "/Frame_Processing/DINO"
        if not os.path.exists(f"{PATH}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
            warnings.warn("GroundingDINO not available. Downloading Grounding DINO GIT repository")
            os.system(f"""git clone https://github.com/IDEA-Research/GroundingDINO.git""")
            os.system(f"""mv GroundingDINO {PATH}/GroundingDINO""")
            os.system(f"""git checkout -q {PATH}/GroundingDINO""")
            os.system(f"""pip install -q -e {PATH}/GroundingDINO""")
            warnings.warn("GroundingDINO installed successfully")
        if not os.path.exists(f"{HOME}/weights/groundingdino_swint_ogc.pth"):
            warnings.warn("GroundingDINO weights not available, downloading")
            os.system(f"""wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth""")
            os.system(f"""mv groundingdino_swint_ogc.pth {HOME}/weights/groundingdino_swint_ogc.pth""")
            warnings.warn("GroundingDINO weight download complete")
        sys.path.append(f"{PATH}/GroundingDINO")
        from groundingdino.util.inference import Model
        GROUNDING_DINO_CONFIG_PATH = os.path.join(PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        self.device = device


    def enhance_class_name(self,class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s" for class_name in class_names
        ]
    def predict(self, frame, classes=None, box_threshold=None, text_threshold=None):
        if classes is None:
            classes = ["car"]
        if box_threshold is None:
            box_threshold = 0.3
        if text_threshold is None:
            text_threshold = 0.25
        detections = self.grounding_dino_model.predict_with_classes(
            image=frame,
            classes=self.enhance_class_name(class_names=classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections.xyxy = torch.from_numpy(detections.xyxy).to(self.device)
        return detections



