
######## Installation Dataset ############

# from roboflow import Roboflow
# rf = Roboflow(api_key="RkzwodUVxKR7dWuHYpFi")
# project = rf.workspace("object-detection-39pwr").project("broken_defection-5p5pu")
# version = project.version(3)
# dataset = version.download("yolov8")


######## Training ###########

# Load a pretrained YOLOv8 model (or start from scratch)
from ultralytics import YOLO


# Load a pretrained YOLOv8 model (or start from scratch)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Train on your custom dataset
if __name__ == '__main__':
    result = model.train(
        data="E:\\ObjectDetection\\Day2\\broken_defection-3\\data.yaml",  # path to your dataset config
        epochs=5,
        imgsz=640,
        batch=16
    )

