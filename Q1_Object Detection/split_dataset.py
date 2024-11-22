import os
import shutil
import random
data_dir = "open-images-dataset"

images_dir = os.path.join(data_dir, "data")
labels_dir = os.path.join(data_dir, "labels")

# Create output directories for train/val splits
train_images_dir = os.path.join(data_dir, "train", "images")
train_labels_dir = os.path.join(data_dir, "train", "labels")
val_images_dir = os.path.join(data_dir, "val", "images")
val_labels_dir = os.path.join(data_dir, "val", "labels")

# Make directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get all image filenames (without extension)
image_files = [f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".jpg")]

# Shuffle and split filenames into train/val sets
random.seed(42)  # Set seed for reproducibility
random.shuffle(image_files)

split_ratio = 0.8  # 80% train, 20% val
train_count = int(split_ratio * len(image_files))

train_files = image_files[:train_count]
val_files = image_files[train_count:]

# Helper function to copy files
def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for file in file_list:
        shutil.copy(os.path.join(src_img_dir, f"{file}.jpg"), dst_img_dir)
        shutil.copy(os.path.join(src_lbl_dir, f"{file}.txt"), dst_lbl_dir)

# Copy train and val files to respective folders
copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
copy_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

print("Dataset split completed successfully!")