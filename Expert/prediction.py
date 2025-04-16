import os
from PIL import Image
import torch
import numpy as np
import cv2
import argparse

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

# Directories
# Dataset directory
parser = argparse.ArgumentParser(description="Dataset configuration")
parser.add_argument('--root', type=str, required=True, help='Root path of the datasets')
parser.add_argument('--output', type=str, default='../Expert_Labels', help='Directory to save the results')
args = parser.parse_args()

# Use parsed arguments
root = args.root
results_dir = args.output
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
datasets = {
    "ddsm": os.path.join(root, "DDSM/VOC2007/JPEGImages/"),
    "inb": os.path.join(root, "INBreast/VOC2007/JPEGImages/"),
    "rsna": os.path.join(root, "RSNA/VOC2007/JPEGImages/"),
}

# Load model
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)
pretrained_pth = 'pretrained/biomedparse_v1.pt'
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# Prompts
prompts = ['mal']

# Inference loop for each dataset
for dataset_name, image_dir in datasets.items():
    results_path = os.path.join(results_dir, f"{dataset_name}_results.txt")
    with open(results_path, "w") as f:
        f.write("ImageName,Prompt,MeanConfidence,MaxConfidence,X,Y,W,H\n")
        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(image_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue

            with torch.no_grad():
                pred_mask = interactive_infer_image(model, image, prompts)

            for i, pred in enumerate(pred_mask):
                mask = pred > 0.5
                binary_mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    f.write(f"{img_name},{prompts[i]},0,0,NA,NA,NA,NA\n")
                    continue

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)

                    # Extract the predicted values inside the contour bounding box
                    roi_pred = pred[y:y+h, x:x+w]
                    roi_mask = mask[y:y+h, x:x+w]

                    if roi_mask.sum() == 0:
                        continue

                    # Confidence only inside the predicted area of the contour box
                    bbox_confidences = roi_pred[roi_mask]
                    # print(f"bbox_confidences: {bbox_confidences}")
                    mean_conf = bbox_confidences.mean()
                    max_conf = bbox_confidences.max()

                    f.write(f"{img_name},{prompts[i]},{mean_conf:.4f},{max_conf:.4f},{x},{y},{w},{h}\n")

    print(f"Finished processing dataset: {dataset_name}, results saved to {results_path}")