from ultralytics import YOLO

# Load a pretrained YOLOv8 model (or start from scratch)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Train on your custom dataset
model.train(
    data="E:\ObjectDetection\Day1\sewing_defect\data.yaml",  # path to your dataset config
    epochs=50,
    imgsz=640,
    batch=16
)
