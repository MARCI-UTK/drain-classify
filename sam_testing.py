from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import json

# Function that inputs the output and plots image and mask
# Code from freeCodeCamp
def show_output(result_dict, axes=None):
     if axes:
        ax = axes
     else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
     sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
     # Plot for each segment area
     for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))

# Set data directory and load images
data_dir = "./data"
img_files = os.listdir(data_dir)[1:]
print(f"Image files: {img_files}")

# Create results directory if not there
results_path = "./sam-auto_results"
if not os.path.exists(results_path):
    os.mkdir(results_path)

sam_models = {
    "base": ("sam_vit_b_01ec64.pth", "vit_b"),
    "large": ("sam_vit_l_0b3195.pth", "vit_l"),
    "huge": ("sam_vit_h_4b8939.pth", "vit_h")
}

# Going through all possible combinations to test performance
# Measure quality of bounding boxes and masks and measure speed of inference
all_times = {}
for sam_name, sam_model in sam_models.items():

    # Load SAM model original way
    sam_checkpoint = f"./pretrained_weights/{sam_model[0]}"
    model_type = sam_model[1]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    auto_mask = SamAutomaticMaskGenerator(sam)
    print(f"Loaded SAM {sam_name} Model")

    for i, img_file in tqdm(enumerate(img_files)):
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path)

        sam_start = time.time()
        masks = auto_mask.generate(img)
        sam_time = sam_start - time.time()

        all_times[f'sam-auto{sam_name}-{img_file}'] = sam_time

# Save times dictionary to file
with open(f'./{results_path}/all_times.txt', 'w') as f:
    json.dump(all_times, f)
print("Saved all times to text file")