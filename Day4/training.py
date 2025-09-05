from ultralytics import YOLO
import os

# Get current folder (where training.py is located)
current_folder = os.path.dirname(os.path.abspath(__file__))

# Paths
data_yaml = os.path.join(current_folder, "E:\\ObjectDetection\\Day4\\bonding\\data.yaml")  # dataset config must be here
save_dir = os.path.join(current_folder, "results")     # where to save best.pt

# Load YOLOv8 model (start from pretrained)
model = YOLO("E:\\ObjectDetection\\Day4\\results9\\weights\\best.pt")

# Train
if __name__ == '__main__':
    result = model.train(
        data=data_yaml,         # dataset config file (must exist in this folder)
        epochs=50,              # training epochs
        imgsz=640,              # image size
        batch=16,               # batch size
        workers=4,              # number of dataloader workers (adjust for speed)
        # ------Optimize-------
        lr0=0.01,               # initial learning rate
        lrf=0.01,               # final learning rate factor
        momentum=0.937,         # SGD momentum, remember the latest path, stable optimization progress
        weight_decay=0.0005,    # optimizer weight decay
        optimizer="SGD",        # optimizer: 'SGD', 'Adam', 'AdamW'
        dropout=0.0,            # dropout (default 0, you can try 0.1–0.2)
        # patience=20,            # early stopping (if val doesn’t improve after X epochs)
        # ------Start Augmetation-------
        close_mosaic=10,        # disable mosaic after N epochs (helps fine-tuning)
        hsv_h=0.015,            # image HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=10,             # rotation (+/- deg)
        translate=0.1,          # translate (%) 
        scale=0.5,              # scale image
        shear=2.0,              # shear angle
        flipud=0.0,             # vertical flip probability
        fliplr=0.5,             # horizontal flip probability
        mosaic=1.0,             # mosaic augmentation probability
        mixup=0.2,              # mixup augmentation probability
        project=current_folder, # save inside this folder
        name="results"          # results folder name
    )

