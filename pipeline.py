from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from util import save_img_with_mask_and_box, save_img_no_background

# Set data directory and load images
data_dir = "./data"
img_files = os.listdir(data_dir)
print(f"Image files: {img_files}")

# Create results directory if not there
results_path = "./results"
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Load YOLO model
yolo = YOLO('./pretrained_weights/yolov8s.pt')
print(f"Loaded YOLOv8 Model")

# Load SAM model original way
sam_checkpoint = "./pretrained_weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
print(f"Loaded SAM Model")

for i, img_file in tqdm(enumerate(img_files)):
    img_path = os.path.join(data_dir, img_file)
    img = Image.open(img_path)

    yolo_results = yolo.predict(img, save=False, conf=0.25)

    for result in yolo_results:
        boxes = result.boxes  # Boxes object for bbox outputs

    bbox=boxes.xyxy.tolist()[0]
    input_box = np.array(bbox)

    input_img = np.array(img) # Convert to numpy
    predictor.set_image(input_img) # Only running encoder once
    masks, _, _ = predictor.predict( # Retrieving the masks
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    save_img_with_mask_and_box(img, masks, bbox, dpi=192, save_path=f'./results/mask_and_box-{img_file}.png')

    save_img_no_background(img, masks, dpi=192, save_path=f'./results/no_background-{img_file}.png')
