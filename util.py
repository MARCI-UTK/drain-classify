import numpy as np
import matplotlib.pyplot as plt

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