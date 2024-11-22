import fiftyone as fo
import fiftyone.zoo as foz

# Load Open Images V7 dataset with specific classes
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",  # Use 'train' for training data
    label_types=["detections"],  # We only need detection labels
    classes=["Laptop", "Book"],  
    max_samples=200,  

# Save the dataset to local files
dataset.export(
    export_dir="open-images-dataset",  # Directory to save the dataset
    dataset_type=fo.types.YOLOv4Dataset,  # Export format for YOLOv8
    labels_path="labels",  # Directory for labels
    images_path="images"   # Directory for images
)
