import torch 
import pandas as pd
from ultralytics import YOLO

model_here = YOLO('yolov8n.pt')
model_here.train(data='dataset/data.yaml', epochs=50,batch=4, imgsz=640)
metrics = model_here.val()
# Extracting relevant metrics
results = metrics.box  # Box metrics can be accessed directly
losses = metrics.loss   # Loss metrics can be accessed directly

# Calculate the metrics to save
metrics_to_save = {
    'Precision': results.P.mean(),  # Mean Precision
    'Recall': results.R.mean(),      # Mean Recall
    'F1 Score': results.F1.mean(),   # Mean F1 Score
    'mAP@0.5': results.mAP50.mean(),  # Mean mAP at IoU threshold 0.5
    'mAP@0.5:0.95': results.mAP50_95.mean(),  # Mean mAP at IoU thresholds 0.5 to 0.95
    'Box Loss': losses.box.mean(),   # Mean Box Loss
    'Class Loss': losses.cls.mean(),  # Mean Class Loss
    'DFL Loss': losses.dfl.mean(),    # Mean DFL Loss
}

print("Metrics to be saved:")
for key, value in metrics_to_save.items():
    print(f"{key}: {value:.4f}")

# Convert metrics to a pandas DataFrame
metrics_df = pd.DataFrame([metrics_to_save])

# Save metrics to a CSV file
metrics_df.to_csv('training_metrics.csv', index=False)

print("Metrics saved to training_metrics.csv")

# Save the fine-tuned model
model_here.save('fine-tuned-model2.pt')
