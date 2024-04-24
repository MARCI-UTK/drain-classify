from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from util import show_box, show_mask, show_points

def save_img_with_mask_and_box(img, masks, bbox, dpi, save_path):
    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi) 
    ax = fig.add_axes([0, 0, 1, 1]) # Makes the image take up full figure
    ax.axis('off') # Don't want to see the axes
    plt.imshow(img)
    show_mask(masks[0], plt.gca())
    if bbox:
        show_box(np.array(bbox), plt.gca())
    plt.axis('off')
    plt.savefig(save_path)

def save_img_no_background(img, masks, dpi, save_path):
    segmentation_mask = masks[0]
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    white_background = np.ones_like(np.array(img)) * 255
    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + np.array(img) * binary_mask[..., np.newaxis]

    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi) 
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    plt.imshow(new_image.astype(np.uint8))
    plt.axis('off')
    plt.savefig(save_path)

# Set data directory and load images
data_dir = "./data"
img_files = os.listdir(data_dir)[1:]
print(f"Image files: {img_files}")

# Create results directory if not there
results_path = "./yolov9_results"
if not os.path.exists(results_path):
    os.mkdir(results_path)

yolo_models = {
    "large": "yolov9c.pt",
    "xlarge": "yolov9e.pt"
}

sam_models = {
    "base": ("sam_vit_b_01ec64.pth", "vit_b"),
    "large": ("sam_vit_l_0b3195.pth", "vit_l"),
    "huge": ("sam_vit_h_4b8939.pth", "vit_h")
}

# Going through all possible combinations to test performance
# Measure quality of bounding boxes and masks and measure speed of inference
all_times = {}
for yolo_name, yolo_model in yolo_models.items():
    for sam_name, sam_model in sam_models.items():

        # Load YOLO model
        yolo = YOLO(f'./pretrained_weights/{yolo_model}')
        print(f"Loaded YOLOv9 {yolo_name} Model")

        # Load SAM model original way
        sam_checkpoint = f"./pretrained_weights/{sam_model[0]}"
        model_type = sam_model[1]
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        predictor = SamPredictor(sam)
        print(f"Loaded SAM {sam_name} Model")

        for i, img_file in tqdm(enumerate(img_files)):
            img_path = os.path.join(data_dir, img_file)
            img = Image.open(img_path)

            # yolo time inference
            yolo_start = time.time()
            yolo_results = yolo.predict(img, save=True,  conf=0.25)
            yolo_time = time.time() - yolo_start

            for result in yolo_results:
                boxes = result.boxes  # Boxes object for bbox outputs
            
            if len(boxes.xyxy.tolist()) > 0:
                bbox = boxes.xyxy.tolist()[0]
                input_box = np.array(bbox)
            else:
                bbox = None

            input_img = np.array(img) # Convert to numpy

            sam_start = time.time()
            predictor.set_image(input_img) # Only running encoder once
            if bbox: # If there is a bounding box
                masks, _, _ = predictor.predict( # Retrieving the masks
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
            else:
                masks, _, _ = predictor.predict( # Retrieving the masks
                    point_coords=None,
                    point_labels=None,
                    box=None,
                    multimask_output=False,
                )
            sam_time = time.time() - sam_start

            total_time = yolo_time + sam_time
            all_times[f'yolo{yolo_name}-sam{sam_name}-{img_file}'] = (yolo_time, sam_time, total_time)

            save_img_with_mask_and_box(img, masks, bbox, dpi=192, save_path=f'./yolov9_results/{yolo_name}-{sam_name}-{img_file}-mab.png')

            save_img_no_background(img, masks, dpi=192, save_path=f'./yolov9_results/{yolo_name}-{sam_name}-{img_file}-noback.png')

# Save times dictionary to file
with open('./yolov9_results/all_times.txt', 'w') as f:
    json.dump(all_times, f)
print("Saved all times to text file")