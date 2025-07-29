
######## Installation Dataset ############

# from roboflow import Roboflow
# rf = Roboflow(api_key="RkzwodUVxKR7dWuHYpFi")
# project = rf.workspace("object-detection-39pwr").project("broken_defection-5p5pu")
# version = project.version(2)
# dataset = version.download("yolov8")

######## Training ###########

from ultralytics import YOLO

# Load a pretrained YOLOv8 model (or start from scratch)
def train():
    model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.
    model.train(
        data="E:\\ObjectDetection\\Day2\\broken_defection-2\\data.yaml",  # path to your dataset config
        epochs=50,
        imgsz=640,
        batch=16

        # Lr0=0.001,
        # weigh_decay=0.0005,
        # dropout=0.2,
        # patience=15
    )
# Train on your custom dataset
if __name__ == "__main__":
    train()
