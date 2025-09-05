import os
import random
import shutil

# Set paths
dataset_path = "E:/ObjectDetection/Day4/bonding"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# New split folders
split_paths = {
    "train_images": os.path.join(dataset_path, "images/train"),
    "val_images": os.path.join(dataset_path, "images/val"),
    "train_labels": os.path.join(dataset_path, "labels/train"),
    "val_labels": os.path.join(dataset_path, "labels/val"),
}

# Create directories
for p in split_paths.values():
    os.makedirs(p, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle
random.shuffle(image_files)

# Train/val split (80/20)
train_size = int(0.8 * len(image_files))
train_files = image_files[:train_size]
val_files = image_files[train_size:]

def move_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        img_src = os.path.join(images_path, file)
        lbl_src = os.path.join(labels_path, file.rsplit(".", 1)[0] + ".txt")

        shutil.copy(img_src, img_dest)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dest)

# Move files
move_files(train_files, split_paths["train_images"], split_paths["train_labels"])
move_files(val_files, split_paths["val_images"], split_paths["val_labels"])

print(f"✅ Done! {len(train_files)} training and {len(val_files)} validation images.")
