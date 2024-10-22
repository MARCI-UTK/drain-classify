import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import torch

from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

YOLO_MODELS = {
    # YOLO 8 
    "yolo8n":   "yolov8n.pt",
    "yolo8s":   "yolov8s.pt",
    "yolo8m":   "yolov8m.pt",
    "yolo8l":   "yolov8l.pt",
    "yolo8x":   "yolov8x.pt",

    # YOLO 10 
    "yolo10n":   "yolov10n.pt",
    "yolo10s":   "yolov10s.pt",
    "yolo10m":   "yolov10m.pt",
    "yolo10l":   "yolov10l.pt",
    "yolo10x":   "yolov10x.pt",

}

SAM_MODELS = {
    # SAM 
    "sam_vitb": ["sam_vit_b_01ec64.pth", "vit_b"],
    "sam_vitl": ["sam_vit_l_0b3195.pth", "vit_l"],
    "sam_vith": ["sam_vit_h_4b8939.pth", "vit_h"],

    # SAM 2
    "sam2_vitt":  "facebook/sam2-hiera-tiny",
    "sam2_vits":  "facebook/sam2-hiera-small",
    "sam2_vitb+": "facebook/sam2-hiera-base-plus",
    "sam2_vitl":  "facebook/sam2-hiera-large",

    # SAM 2.1
    "sam2.1_vitt":  "facebook/sam2.1-hiera-tiny",
    "sam2.1_vits":  "facebook/sam2.1-hiera-small",
    "sam2.1_vitb+": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_vitl":  "facebook/sam2.1-hiera-large",
}


class YOLOConfig(object):
    def __init__(self):
        self.pretrained_weights_dir = "./pretrained_models"

        self.yolo_model = "yolo10m"
        self.yolo_pretrained = os.path.join(self.pretrained_weights_dir, YOLO_MODELS[self.yolo_model])
        self.yolo = YOLO(self.yolo_pretrained)

        self.sam_model = "sam_vitb"

        # SAM 2 and SAM 2.1
        if "2" in self.sam_model:
            self.sam = SAM2ImagePredictor.from_pretrained(SAM_MODELS[self.sam_model], device=torch.device("cpu"))
        # SAM
        else:
            self.sam_model_config = SAM_MODELS[self.sam_model][1]
            self.sam_pretrained = os.path.join(self.pretrained_weights_dir, SAM_MODELS[self.sam_model][0])
            self.sam = SamPredictor(sam_model_registry[self.sam_model_config](
                checkpoint=self.sam_pretrained)
            )


    def save(self, save_dir):
        config = {
            "yolo_model": self.yolo_model,
            "sam_model": self.sam_model
        }

        save_path = os.path.join(save_dir, "config.json")
        with open(save_path, 'w') as file:
            json.dump(config, file)

class DINOConfig(object):
    def __init__(self):
        self.pretrained_weights_dir = "./pretrained_models"

        self.dino_model = "grounding_dino"
        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        self.detection_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )

        self.sam_model = "sam_vitb"

        # SAM 2 and SAM 2.1
        if "2" in self.sam_model:
            self.sam = SAM2ImagePredictor.from_pretrained(SAM_MODELS[self.sam_model], 
                                                          device=torch.device("cpu"))
        # SAM
        else:
            self.sam_model_config = SAM_MODELS[self.sam_model][1]
            self.sam_pretrained = os.path.join(self.pretrained_weights_dir, SAM_MODELS[self.sam_model][0])
            self.sam = SamPredictor(sam_model_registry[self.sam_model_config](
                checkpoint=self.sam_pretrained)
            )


    def save(self, save_dir):
        config = {
            "dino_model": self.dino_model,
            "sam_model": self.sam_model
        }

        save_path = os.path.join(save_dir, "config.json")
        with open(save_path, 'w') as file:
            json.dump(config, file)

