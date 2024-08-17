from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(data="coco.yaml", epochs=100, imgsz=640, device="mps")
success = model.export()