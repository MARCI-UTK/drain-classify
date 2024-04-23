from ultralytics import SAM, YOLO
from PIL import Image
import os
import tqdm

# Set data directory and load images
data_dir = "./data"
img_files = os.listdir(data_dir)

# Load YOLO model
yolo = YOLO('yolov8n.pt')
yolo_info = yolo.info()
print(f"Loaded YOLOv8 Model; Layers: {yolo_info[0]}, Params: {yolo_info[1]}")

# Load SAM model
sam = SAM('sam_b.pt')
sam_info = sam.info()
print(f"Loaded SAM Model; Layers: {sam_info[0]}, Params: {sam_info[1]}")

for img, i in tqdm(enumerate(img_files)):
    img_path = os.path.join(data_dir, img)
    img = Image.open(img_path)

    yolo_results = yolo.predict(img, save=True, conf=0.25)

    for result in yolo_results:
        boxes = result.boxes  # Boxes object for bbox outputs

    bbox=boxes.xyxy.tolist()[0]

    # sam_results = sam.predict(img, bboxes=None)
