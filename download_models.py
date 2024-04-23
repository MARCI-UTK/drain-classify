from ultralytics import YOLO
import wget

def download_all():
    yolo8_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    sam_models = ['https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth']
    sam_models_names = ['ViT-B', 'ViT-L', 'ViT-H']
        
    for model_name in yolo8_models:
        yolo = YOLO(model_name)
        print(f"Downloaded {model_name}\n")

    for i, model_url in enumerate(sam_models):
        wget.download(model_url)
        print(f"Downloaded SAM {sam_models_names[i]}\n")


download_all()

# Place the models into a new folder named "pretrained_weights"