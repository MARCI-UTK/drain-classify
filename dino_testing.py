import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from util import show_mask

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125]]

def plot_results(img, text, scores, labels, boxes, masks, save_dir, file_name):
    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / 192, img_height / 192), dpi=192)
    # Making sure the image takes up full figure with no axes
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') 

    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{text}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
    show_mask(masks[0], plt.gca())

    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'{file_name}.png'))

def detect_drain(img, text):
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    detection_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    # This returns similar values as an image preprocessor and text tokenizer
    # i.e. 'input_ids', 'attention_mask', 'pixel_values'
    inputs = processor(images=img, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = detection_model(**inputs)

    # postprocess model outputs
    width, height = img.size
    postprocessed_outputs = processor.image_processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=0.3)
    results = postprocessed_outputs[0]
    
    return results

def main(): 
    save_dir = os.path.join('./results/dino_testing', datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # Data directory containing drain images
    data_dir = "./data/raw"
    # The "[1:]" removes a non-image file contained in dir
    img_files = os.listdir(data_dir)[1:]

    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path)
        text = "a surgical drain."

        results = detect_drain(img, text)
        input_box = np.asarray(results['boxes'][0])

        sam = SamPredictor(sam_model_registry["vit_b"](checkpoint="./pretrained_models/sam_vit_b_01ec64.pth"))
        img_np = np.asarray(img)
        sam.set_image(img_np)

        masks, _, _ = sam.predict( 
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )
        
        plot_results(img=img, text=text,
                    scores=results['scores'].tolist(), 
                    labels=results['labels'].tolist(), 
                    boxes=results['boxes'].tolist(),
                    masks=masks, 
                    save_dir=save_dir,
                    # Removing extension from file name
                    file_name=img_file[:-4],)

if __name__ == "__main__":
    main()
