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

def test(yolo_version="8"):
    # Set YOLO model version; Options: [5, 8, 10]
    yolo_version = "8"

    # Data directory containing drain images
    data_dir = "./raw_data"
    # The "[1:]" removes a non-image file contained in dir
    img_files = os.listdir(data_dir)[1:]
    print(f"Image files: {img_files}")

    # Create results directory if not there
    results_path = f"./mask_results/yolov{yolo_version}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Dict used for accessing different YOLO models in same run
    yolo_models = {
        "nano": f"yolov{yolo_version}n.pt",
        "small": f"yolov{yolo_version}s.pt",
        "medium": f"yolov{yolo_version}m.pt",
        # "large": f"yolov{yolo_version}l.pt",
        # "xlarge": f"yolov{yolo_version}x.pt"
    }

    sam_models = {
        "base": ("sam_vit_b_01ec64.pth", "vit_b"),
        # "large": ("sam_vit_l_0b3195.pth", "vit_l"),
        # "huge": ("sam_vit_h_4b8939.pth", "vit_h")
    }

    # Going through all possible combinations to test performance
    # Measure quality of bounding boxes and masks and measure speed of inference
    all_times = {}
    for yolo_name, yolo_model in yolo_models.items():
        for sam_name, sam_model in sam_models.items():

            # Load YOLO model
            yolo = YOLO(f'./pretrained_weights/{yolo_model}')
            print(f"Loaded {yolo_name} model")

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
                predictor.set_image(input_img) 
            
                if bbox:
                    # Retrieving the masks
                    masks, _, _ = predictor.predict( 
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                else:
                    # Retrieving the masks with no bounding box
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=None,
                        multimask_output=False,
                    )

                # Total SAM inference time    
                sam_time = time.time() - sam_start

                total_time = yolo_time + sam_time
                all_times[f'yolo{yolo_name}-sam{sam_name}-{img_file}'] = (yolo_time, sam_time, total_time)

                save_img_with_mask_and_box(img, masks, bbox, dpi=192, save_path=f'./{results_path}/{yolo_name}-{sam_name}-{img_file}-mab.png')

                save_img_no_background(img, masks, dpi=192, save_path=f'./{results_path}/{yolo_name}-{sam_name}-{img_file}-noback.png')

    # Save times dictionary to file
    with open(f'./{results_path}/all_times.txt', 'w') as f:
        json.dump(all_times, f)
    print("Saved all times to text file")

if __name__ == "__main__":
    test()