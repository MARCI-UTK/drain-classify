import os
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def save_img_bbox_and_mask(img, masks, bbox, dpi=192, save_path=None, file_name=None):
    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi) 
    
    # Makes the image take up full figure
    ax = fig.add_axes([0, 0, 1, 1])
    # Don't want to see the axes
    ax.axis('off') 

    plt.imshow(img)

    if masks:
        show_mask(masks[0], plt.gca())
    if bbox:
        show_box(np.array(bbox), plt.gca())
    
    # Making sure figure is just image
    plt.axis('off')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    plt.savefig(os.path.join(save_path, f'{file_name}.png'))

def save_img_masked_out(img, masks, dpi=192, save_path=None, file_name=None):
    segmentation_mask = masks[0]
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    white_background = np.ones_like(np.array(img)) * 255
    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + np.array(img) * binary_mask[..., np.newaxis]

    img_width, img_height = img.size
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi) 
    # Makes image take up full figure
    ax = fig.add_axes([0, 0, 1, 1])
    # Only want image to be displayed
    ax.axis('off')

    plt.imshow(new_image.astype(np.uint8))
    plt.axis('off')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    plt.savefig(os.path.join(save_path, f'{file_name}.png'))

def display_images(img_dir):
    img_files = os.listdir(img_dir)[1:]
    logging.info(f"{img_files}")
    images = []
    for img_file in img_files:
        image = Image.open(os.path.join(img_dir, img_file))
        images.append(image)
    
    cols = 6 
    rows = 3  
    
    # Create a figure to hold subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))

    # Flatten the axis array for easy iteration if there are multiple rows/cols
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]
    
    for i, img in enumerate(images):
        axs[i].imshow(img)          
        axs[i].axis('off')          
    
    # Hide any unused subplots (if any)
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'showcase.png'))