from clearml import Task
task = Task.init(project_name="drain", task_name="mask-testing") 

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import json
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from util import save_img_no_background, save_img_with_mask_and_box

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

    # SAM 2.1
    "sam2.1_vitt":  "facebook/sam2.1-hiera-tiny",
    "sam2.1_vits":  "facebook/sam2.1-hiera-small",
    "sam2.1_vitb+": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_vitl":  "facebook/sam2.1-hiera-large",
}


class Config(object):
    def __init__(self):
        self.pretrained_weights_dir = "./pretrained_models"

        self.yolo_model = "yolo10m"
        self.yolo_pretrained = os.path.join(self.pretrained_weights_dir, YOLO_MODELS[self.yolo_model])
        self.yolo = YOLO(self.yolo_pretrained)

        self.sam_model = "sam2.1_vitb+"

        # SAM 2.1
        if "2" in self.sam_model:
            self.sam = SAM2ImagePredictor.from_pretrained(SAM_MODELS[self.sam_model], device=torch.device("cpu"))
        # SAM
        else:
            self.sam = SamPredictor(sam_model_registry[self.sam_model_config](checkpoint=self.sam_pretrained))

    def save(self, save_dir):
        config = {
            "yolo_model": self.yolo_model,
            "sam_model": self.sam_model
        }

        save_path = os.path.join(save_dir, "config.txt")
        with open(save_path, 'w') as file:
            json.dump(config, file)
            

def run(yolo, sam, yolo_name, sam_name, save_dir):
    # Data directory containing drain images
    data_dir = "./data/raw"
    # The "[1:]" removes a non-image file contained in dir
    img_files = os.listdir(data_dir)[1:]
    
    times = {}
    for i, img_file in tqdm(enumerate(img_files)):
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path)

        # yolo time inference
        yolo_start = time.time()
        yolo_results = yolo.predict(img, save=True, conf=0.25, 
                                    project=save_dir, name="detect", exist_ok=True)
        yolo_time = time.time() - yolo_start

        for result in yolo_results:
            # Boxes object for bbox outputs
            boxes = result.boxes  
        
        if len(boxes.xyxy.tolist()) > 0:
            bbox = boxes.xyxy.tolist()[0]
            input_box = np.array(bbox)
        else:
            bbox = None

        input_img = np.array(img) 

        # Used to calculate inference time
        sam_start = time.time()

        # Only running encoder once
        sam.set_image(input_img) 
    
        if bbox:
            # Retrieving the masks
            masks, _, _ = sam.predict( 
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
        else:
            # Retrieving the masks with no bounding box
            masks, _, _ = sam.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )

        # Total SAM inference time and total time 
        sam_time = time.time() - sam_start
        total_time = yolo_time + sam_time

        times["yolo"] = yolo_time
        times["sam"] = sam_time
        times["total"] = total_time

        save_img_with_mask_and_box(img, masks, bbox, dpi=192, save_path=os.path.join(save_dir, f'{yolo_name}-{sam_name}-{img_file}-mab.png'))
        save_img_no_background(img, masks, dpi=192, save_path=os.path.join(save_dir, f'{yolo_name}-{sam_name}-{img_file}-noback.png'))

    # Save times dictionary to file
    with open(os.path.join(save_dir, "times.txt"), 'w') as f:
        json.dump(times, f)

if __name__ == "__main__":

    config = Config()
    task.connect(config, name="config")

    # Set save_dir and ensure it exists
    save_dir = os.path.join('./results/mask_testing', datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config.save(save_dir)

    run(
        yolo=config.yolo,
        sam=config.sam,
        yolo_name=config.yolo_model,
        sam_name=config.sam_model,
        save_dir=save_dir,
    )
