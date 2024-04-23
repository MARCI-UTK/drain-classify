from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from util import show_box, show_mask, show_points

def save_img_with_mask_and_box(img, masks, bbox, dpi, save_path):
    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi) 
    ax = fig.add_axes([0, 0, 1, 1]) # Makes the image take up full figure
    ax.axis('off') # Don't want to see the axes
    plt.imshow(img)
    show_mask(masks[0], plt.gca())
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
img_files = os.listdir(data_dir)
print(f"Image files: {img_files}")

# Create results directory if not there
results_path = "./results"
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Load YOLO model
yolo = YOLO('./pretrained_weights/yolov8n.pt')
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


# Load SAM model with Ultralytics
# overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
# predictor = SAMPredictor(overrides=overrides)

# Predict with Ultralytics SAM model
# predictor.set_image(img)  # set with image file
# results = predictor(bboxes=input_box)

