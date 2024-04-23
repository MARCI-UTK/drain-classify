from ultralytics import SAM
from PIL import Image
import os
import tqdm

# Set data directory and load images
data_dir = "./data"
img_files = os.listdir(data_dir)

# Load YOLO

# Load SAM model
model = SAM('sam_b.pt')
info = model.info()
print(f"Loaded SAM Model; Layers: {info[0]}, Params: {info[1]}")

for img, i in tqdm(enumerate(img_files)):
    img_path = os.path.join(data_dir, img)
    img = Image.open(img_path)

    results = model.predict(img, bboxes=None)
