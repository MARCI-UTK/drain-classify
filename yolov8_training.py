from ultralytics import YOLO

model = YOLO('./pretrained_weights/yolov8n.pt')
results = model.train(data="./DC/data.yaml", epochs=5, imgsz=640, device="mps")
success = model.export()