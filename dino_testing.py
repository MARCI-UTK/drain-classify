# from clearml import Task
# task = Task.init(project_name="drain", task_name="dino-testing") 

import os
import torch
import numpy as np

from PIL import Image
from datetime import datetime

from configs import DINOConfig
from util import save_img_masked_out, save_img_bbox_and_mask

def detect_drain(img, text, processor, detection_model):

    # This returns similar values as an image preprocessor and text tokenizer
    # i.e. 'input_ids', 'attention_mask', 'pixel_values'
    inputs = processor(images=img, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = detection_model(**inputs)

    # postprocess model outputs
    width, height = img.size
    postprocessed_outputs = processor.image_processor.post_process_object_detection(
        outputs,
        target_sizes=[(height, width)],
        threshold=0.3
    )
    results = postprocessed_outputs[0]
    
    return results


def test(processor, detection_model, sam, dino_name, sam_name, save_dir):
    # Data directory containing drain images
    data_dir = "./data/raw"
    # The "[1:]" removes a non-image file contained in dir
    img_files = os.listdir(data_dir)[1:]

    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path)
        text = "a surgical drain."

        results = detect_drain(img, text, processor, detection_model)
        # Picking the highest confidence bbox
        input_box = np.asarray(results['boxes'][0])

        sam.set_image(np.array(img))

        masks, _, _ = sam.predict( 
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )

        file_name = f'{dino_name}-{sam_name}-{img_file[:-4]}'

        save_img_bbox_and_mask(img, masks[0], input_box,
                               save_path=os.path.join(save_dir, 'bbox_and_mask'),
                               file_name=file_name)
        
        save_img_masked_out(img, masks[0],
                            save_path=os.path.join(save_dir, 'masked_out'),
                            file_name=file_name)


def main(): 
    config = DINOConfig()
    # task.connect(config, name="config")

    save_dir = os.path.join('./results/dino_testing', datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    config.save(save_dir)

    test(
        processor=config.processor,
        detection_model=config.detection_model,
        sam=config.sam,
        dino_name=config.dino_model,
        sam_name=config.sam_model,
        save_dir=save_dir
    )
    

if __name__ == "__main__":
    main()
