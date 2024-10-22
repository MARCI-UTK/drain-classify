# from clearml import Task
# task = Task.init(project_name="drain", task_name="yolo-testing") 

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np

from PIL import Image
from tqdm import tqdm
from datetime import datetime

from configs import YOLOConfig
from util import save_img_bbox_and_mask, save_img_masked_out


def test(yolo, sam, yolo_name, sam_name, save_dir):
    # Data directory containing drain images
    data_dir = "./data/raw"
    # The "[1:]" removes a non-image file contained in dir
    img_files = os.listdir(data_dir)[1:]

    for i, img_file in tqdm(enumerate(img_files)):
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path)

        # Results is a Results object from ultralytics
        yolo_results = yolo.predict(img, save=True, conf=0.25, 
                                    project=save_dir, name="detect", 
                                    exist_ok=True)
        # Getting single Results object in list with length 1
        yolo_results = yolo_results[0]
        # List of Boxes objects containing bounding boxes
        boxes = yolo_results.boxes

        # Creating a list of bounding boxes    
        bboxs = []        
        if len(boxes.xyxy.tolist()) > 0:
            bbox = np.array(boxes.xyxy.tolist()[0])
            bboxs.append(bbox)

        # Setting SAM to correct image
        sam.set_image(np.array(img)) 
    
        # Checking if element in bboxs and just choosing first bbox
        if len(bboxs) > 0:
            masks, _, _ = sam.predict( 
                point_coords=None,
                point_labels=None,
                box=bboxs[0],
                multimask_output=False,
            )
        # If no bbox, then try segmentation on full image
        else:
            masks, _, _ = sam.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )

        file_name = f'{yolo_name}-{sam_name}-{img_file[:-4]}'

        # Saves an image showing top-1 bounding box and mask
        if len(bboxs) > 0:
            save_img_bbox_and_mask(img, masks[0], bboxs[0], 
                                    save_path=os.path.join(save_dir, 'bbox_and_mask'), 
                                    file_name=file_name)
        else:
            save_img_bbox_and_mask(img, masks[0], bbox=None, 
                                    save_path=os.path.join(save_dir, 'bbox_and_mask'), 
                                    file_name=file_name)

        # Saves image with masked, white background using top-1 segmentation mask
        save_img_masked_out(img, masks[0], 
                            save_path=os.path.join(save_dir, 'masked_out'), 
                            file_name=file_name)


def main():
    config = YOLOConfig()
    # task.connect(config, name="config")

    # Set save_dir and ensure it exists
    save_dir = os.path.join('./results/yolo_testing', datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    config.save(save_dir)

    test(
        yolo=config.yolo,
        sam=config.sam,
        yolo_name=config.yolo_model,
        sam_name=config.sam_model,
        save_dir=save_dir,
    )

if __name__ == "__main__":
    main()