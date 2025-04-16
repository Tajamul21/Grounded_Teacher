import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import detectron2.utils.comm as comm  
from detectron2.config import get_cfg  
from detectron2.utils.logger import setup_logger  
from detectron2.engine import DefaultTrainer  
from detectron2.checkpoint import DetectionCheckpointer  
from detectron2.data import build_detection_test_loader  
from detectron2.evaluation import inference_on_dataset  
from GT.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from detectron2.data import MetadataCatalog
from GT.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from GT.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from GT.modeling.proposal_generator.rpn import PseudoLabRPN
from GT.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from GT.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
import xml.etree.ElementTree as ET
import json
from scipy.interpolate import interp1d
import pandas as pd
from openpyxl import Workbook
from datetime import datetime


from GT.data.datasets.builtin import(
    register_all_pascal_voc,
    _root
)

from GT import add_cat_config

def full_prediction(predictions, meta, score_threshold=0.001):
    ann_path = os.path.join(meta.dirname, "Annotations/")
    
    formatted_predictions = []

    for cls, preds in predictions.items():
        
        
        for pred in preds:
            image_id, score, xmin, ymin, xmax, ymax = pred.split()
            score = float(score)
            xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])

            # ✅ Filter by score threshold
            if score < score_threshold:
                continue

            # ✅ Create new entry if not already present
            existing_entry = next(
                (entry for entry in formatted_predictions if entry["file_name"] == image_id),
                None
            )

            if existing_entry is None:
                existing_entry = {
                    "file_name": image_id,
                    "pred": {
                        "boxes": [],
                        "scores": [],
                        "cls_pred": []
                    },
                    "target": {
                        "boxes": []
                    }
                }
                formatted_predictions.append(existing_entry)

            # ✅ Append prediction details
            if cls != 0:
                existing_entry["pred"]["boxes"].append([xmin, ymin, xmax, ymax])
                existing_entry["pred"]["scores"].append(score)
                existing_entry["pred"]["cls_pred"].append(cls)

            # ✅ Load ground truth boxes only if not already loaded
            if not existing_entry["target"]["boxes"]:
                new_ann_path = os.path.join(ann_path, f"{image_id}.xml")
                if os.path.exists(new_ann_path):
                    tree = ET.parse(new_ann_path)
                    root = tree.getroot()
                    
                    for obj in root.findall("object"):
                        bbox = obj.find("bndbox")
                        x_min = float(bbox.find("xmin").text)
                        y_min = float(bbox.find("ymin").text)
                        x_max = float(bbox.find("xmax").text)
                        y_max = float(bbox.find("ymax").text)
                        existing_entry["target"]["boxes"].append([x_min, y_min, x_max, y_max])

    return formatted_predictions

def create_excel_file(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        wb = Workbook()
        ws = wb.active
        ws.append(["model_iter", "fpi", "tpr", "froc_values", "fpi_levels"])
        wb.save(file_path)
        print(f"Excel file created: {file_path}")

def update_excel_file(file_path, model_iter, fpi, tpr, froc_values, fpi_levels):
    df = pd.read_excel(file_path)
    new_entry = {"model_iter": str(model_iter), "fpi": str(fpi), "tpr": str(tpr), "froc_values": str(froc_values), "fpi_levels": str(fpi_levels)}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(file_path, index=False)
    print(f"Updated Excel file: {file_path}")
    
def filter_best_predictions(pred_map, score_threshold=0.001):
    best_preds = {}

    for cls, preds in pred_map.items():
        for pred in preds:
            parts = pred.strip().split()
            if len(parts) < 6:
                continue
            image_id = parts[0]
            score = float(parts[1])
            if score < score_threshold:
                continue

            if image_id not in best_preds or score > best_preds[image_id][0]:
                best_preds[image_id] = (score, cls, pred.strip())

    filtered_map = {}
    for _, (score, cls, pred_str) in best_preds.items():
        if cls not in filtered_map:
            filtered_map[cls] = []
        filtered_map[cls].append(pred_str)

    return filtered_map




def full_prediction_unique(prediction_map, meta, score_threshold=0.001):
    ann_path = os.path.join(meta.dirname, "Annotations/")
    print(f"Gt Path: {ann_path}")
    
    filtered_map = filter_best_predictions(prediction_map, score_threshold)

    formatted_predictions = []
    for cls, preds in filtered_map.items():
        for pred_str in preds:
            parts = pred_str.strip().split()
            if len(parts) < 6:
                continue  

            image_id = parts[0]
            score = float(parts[1])
            xmin, ymin, xmax, ymax = map(float, parts[2:6])
            cls = int(cls)

            box = [xmin, ymin, xmax, ymax] if cls != 0 else []
            score = score if cls != 0 else [0]

            new_ann_path = os.path.join(ann_path, f"{image_id}.xml")
            target_boxes = []
            if os.path.exists(new_ann_path):
                tree = ET.parse(new_ann_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    x_min = float(bbox.find("xmin").text)
                    y_min = float(bbox.find("ymin").text)
                    x_max = float(bbox.find("xmax").text)
                    y_max = float(bbox.find("ymax").text)
                    target_boxes.append([x_min, y_min, x_max, y_max])

            formatted_predictions.append({
                "file_name": image_id,
                "pred": {
                    "boxes": [box] if cls != 0 else [],
                    "scores": [score] if cls != 0 else [0],
                    "cls_pred": [cls]
                },
                "target": {
                    "boxes": target_boxes
                }
            })

    return formatted_predictions

def save_predictions(predictions, filename="predictions.txt"):
    with open(filename, "w") as f:
        for cls, pred_list in predictions.items():
            for pred in pred_list:
                f.write(f"{cls}: {pred}\n")
    print(f"Predictions saved to {filename}")    
    
    
def plot_froc_curve(fpi_values, tpr_values, save_path, CHECKPOINT):
    plt.figure(figsize=(10, 6))

    # Handle empty or invalid values
    if not fpi_values or not tpr_values:
        print("Warning: Empty FPI or TPR values, skipping FROC plot.")
        return
    
    # Convert lists to numpy arrays for safer calculations
    fpi_values = np.array(fpi_values)
    tpr_values = np.array(tpr_values)

    # Ensure valid numerical values
    fpi_values = np.nan_to_num(fpi_values, nan=0.0, posinf=0.0, neginf=0.0)
    tpr_values = np.nan_to_num(tpr_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Get valid limits
    max_fpi = np.ceil(max(fpi_values)) if len(fpi_values) > 0 else 1
    max_tpr = np.ceil(max(tpr_values)) if len(tpr_values) > 0 else 1

    # Plot the FROC curve
    plt.xlim(0, max_fpi)
    plt.ylim(0, max_tpr)
    plt.plot(fpi_values, tpr_values, 'b-', marker='o')
    
    plt.xlabel('False Positives per Image (FPI)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('FROC Curve')
    plt.grid(True)

    # Save the plot
    Checkpoint_name = CHECKPOINT.replace(".pth", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{Checkpoint_name}_froc_curve_{timestamp}.png"
    save_file = os.path.join(save_path, filename)

    plt.savefig(save_file)
    print(f"FROC curve saved at {save_file}")

    plt.close()
    
def setup_cfg(args):  
    cfg = get_cfg()
    add_cat_config(cfg)  
    cfg.merge_from_file(args.config_file)  
    cfg.merge_from_list(args.opts)  
    cfg.MODEL.WEIGHTS = args.model_path 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Set a custom testing threshold  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=2 
    cfg.freeze()  
    return cfg  

import numpy as np

def calculate_froc(predictions, fpi_levels = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1], iou_threshold=0.5):
    all_scores, all_is_tp, total_gt = [], [], 0

    for data in predictions:
        pred_boxes = data["pred"]["boxes"]
        pred_scores = data["pred"]["scores"]
        gt_boxes = data["target"]["boxes"]
        total_gt += len(gt_boxes)

        all_scores.extend(pred_scores)
        all_is_tp.extend([1 if len(gt_boxes) > 0 else 0] * len(pred_scores))

    if len(all_scores) == 0:
        print("Error: No scores found.")
        return [], []

    all_scores = np.array(all_scores)
    all_is_tp = np.array(all_is_tp)
    sort_idx = np.argsort(-all_scores) 

    sorted_tp = all_is_tp[sort_idx]
    cumulative_tp = np.cumsum(sorted_tp)
    cumulative_fp = np.cumsum(1 - sorted_tp)

    fpi = cumulative_fp / len(predictions)
    tpr = cumulative_tp / total_gt if total_gt > 0 else np.zeros_like(cumulative_tp)

    if len(fpi) == 0 or len(tpr) == 0:
        print("Error: FPI or TPR is empty.")
        return [], []

    interpolator = interp1d(fpi, tpr, bounds_error=False, fill_value="extrapolate")

    return fpi.tolist(), tpr.tolist(), interpolator(fpi_levels).tolist(), fpi_levels



def get_confmat(pred_list, threshold = 0.1):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[2]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        
        scores = np.array(pred['scores'])
        boxes = np.array(pred['boxes'])

        if len(scores) != len(boxes):
            continue  # or optionally log/debug this case

        select_mask = scores > threshold
        pred_boxes = boxes[select_mask] if len(boxes) > 0 else np.array([])

        out_array = np.zeros((4))
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred in pred_boxes:
                if true_positive(gt_box, pred):
                    add_tp = True
                else:
                    new_preds.append(pred)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1  # TP
            else:
                out_array[3] += 1  # FN

        out_array[2] = len(pred_boxes)  # Remaining predictions = FP
        conf_mat += out_array

    return conf_mat
    


def calc_froc(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1], num_thresh = 1000):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat
    
    sensitivity = np.zeros((num_thresh)) #recall
    specificity = np.zeros((num_thresh)) #presicion
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
        if((conf_mat[0]+conf_mat[2])==0):
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[2])

    senses_req = []
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            if f/num_images < fp_req:
                senses_req.append(sensitivity[i-1])
                print(fp_req, sensitivity[i-1], thresholds[i], get_confmat(pred_data, thresholds[i]))
                break
    return senses_req, fps_req, sensitivity, specificity
    

    
    
def main(args,output,CHECKPOINT,EXCEL_FILE_PATH):
    cfg = setup_cfg(args,)
    target=cfg.DATASETS.TEST[0]
    print(f"TARGET: {target}")
    if target not in MetadataCatalog.list():
        from GT.data.datasets.builtin import register_all_pascal_voc, _root
        register_all_pascal_voc(_root)
    meta = MetadataCatalog.get(target)
    model = DefaultTrainer.build_model(cfg)  
      
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = PascalVOCDetectionEvaluator(target)  
    val_loader = build_detection_test_loader(cfg,target) 
    evaluator.reset() 
    print("Starting Pascal VOC evaluation on the target dataset.")  
    inference_results = inference_on_dataset(model, val_loader, evaluator) 
    checkpoint_name = CHECKPOINT.replace(".pth", "")
    save_predictions(evaluator._predictions, f"{output}/GTLabels_{checkpoint_name}.txt")
    predictions=full_prediction_unique(evaluator._predictions,meta)
    with open(f"{output}/GTLabels_{checkpoint_name}.json", "w") as f:
        json.dump(predictions, f, indent=4)
    fpi,tpr,froc_values, fpi_levels = calculate_froc(predictions)
    update_excel_file(EXCEL_FILE_PATH, CHECKPOINT, fpi, tpr, froc_values, fpi_levels)
    
    plot_froc_curve(fpi_levels, froc_values,output,CHECKPOINT)
    print(get_confmat(predictions))
    return inference_results,  

if __name__ == "__main__":  
    import argparse  
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model on a Pascal VOC dataset")
    parser.add_argument("--setting", help="Experiment setting name", default="ddsm2rsna")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs", default=[], nargs='+')
    parser.add_argument("--root", help="Root output directory", default=None)

    args = parser.parse_args()

    # If root not provided explicitly, use default based on setting
    if args.root is None:
        args.root = f"output/{args.setting}"
    
    args.config_file=f"{args.root}/config.yaml"
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    OUTPUT=f"{args.root}/FrocResults" 
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    EXCEL_FILE_PATH = f"{OUTPUT}/{args.setting}.xlsx"
    create_excel_file(EXCEL_FILE_PATH)
    
    # CHECKPOINTS = ["model_0019999.pth","model_0019999.pth"] # for selected checkpoint
    CHECKPOINTS = sorted(glob.glob(os.path.join(args.root, "*.pth")))  # for all selected checkpoint
    for CHECKPOINT in CHECKPOINTS:
        print(f"Checkpoint Loaded:{CHECKPOINT}")
        CHECKPOINT=CHECKPOINT.replace(f"{args.root}/",'')
        args.model_path = f"{args.root}/{CHECKPOINT}"
        print(f"Setting: {args.setting}")
        print(f"Model Path: {args.model_path}")
        print(f"Config File: {args.config_file}")
        setup_logger()  
        
        main(args,OUTPUT,CHECKPOINT,EXCEL_FILE_PATH)  