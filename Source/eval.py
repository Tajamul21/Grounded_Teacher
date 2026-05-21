# coding:utf-8
# python eval.py --dataset_t voc_2007_test --load_name "/DATA/Tawheed/SFDA/Grounded_Teacher/Source/output/vgg16/brats/lg_adv_session_1_epoch_6_step_10000.pth"

from __future__ import absolute_import, division, print_function
import json, os, numpy as np, pprint, xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn

import _init_paths
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import FocalLoss, EFocalLoss
from model.utils.parser_func import parse_args, set_dataset_args
from model.faster_rcnn.vgg16_adv import vgg16

from PIL import Image

import logging
logging.basicConfig(
    filename='logs/step3.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------------------------------
# FROC FUNCTIONS
# -------------------------------------------------------------------------

def calculate_froc(predictions, fpi_levels=np.linspace(0, 5, 50), iou_threshold=0.5):

    all_scores, all_is_tp, total_gt = [], [], 0

    for data in predictions:
        pred_scores = data["pred"]["scores"]
        gt_boxes = data["target"]["boxes"]

        total_gt += len(gt_boxes)

        for score in pred_scores:
            all_scores.append(score)

        all_is_tp.extend([1 if len(gt_boxes) > 0 else 0] * len(pred_scores))

    if len(all_scores) == 0:
        print("Error: No scores found.")
        return [], [], [], []

    sort_idx = np.argsort(-np.array(all_scores))

    sorted_tp = np.array(all_is_tp)[sort_idx]
    cumulative_tp = np.cumsum(sorted_tp)
    cumulative_fp = np.cumsum(1 - sorted_tp)

    fpi = cumulative_fp / len(predictions)
    tpr = cumulative_tp / total_gt if total_gt > 0 else np.zeros_like(cumulative_tp)

    interpolator = interp1d(fpi, tpr, bounds_error=False, fill_value=(0, 1))

    return fpi, tpr, interpolator(fpi_levels).tolist(), fpi_levels.tolist()


def plot_froc_curve(fpi_values, tpr_values):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, max(fpi_values) if len(fpi_values) else 1)
    plt.ylim(0, max(tpr_values) if len(tpr_values) else 1)
    plt.plot(fpi_values, tpr_values, 'b-', marker='o')
    plt.xlabel('False Positives per Image (FPI)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('FROC Curve')
    plt.grid(True)
    plt.savefig("froc_curve.png")
    plt.close()


# -------------------------------------------------------------------------
# PREDICTION PARSER
# -------------------------------------------------------------------------

def full_prediction(predictions, score_threshold=0.001):
    ann_path = "/DATA/Tawheed/Dataset/Natural/VOC_city/VOC2007/Annotations/"

    formatted_predictions = []

    for cls, preds in predictions.items():
        for pred in preds:
            image_id, score, xmin, ymin, xmax, ymax = pred.split()

            score = float(score)
            xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])

            if score < score_threshold:
                continue

            entry = next((x for x in formatted_predictions if x["file_name"] == image_id), None)
            if entry is None:
                entry = {
                    "file_name": image_id,
                    "pred": {"boxes": [], "scores": [], "cls_pred": []},
                    "target": {"boxes": []}
                }
                formatted_predictions.append(entry)

            if cls != 0:
                entry["pred"]["boxes"].append([xmin, ymin, xmax, ymax])
                entry["pred"]["scores"].append(score)
                entry["pred"]["cls_pred"].append(cls)

            if not entry["target"]["boxes"]:
                xml_file = os.path.join(ann_path, image_id.replace(".jpg", ".xml"))
                if os.path.exists(xml_file):
                    root = ET.parse(xml_file).getroot()
                    for obj in root.findall("object"):
                        bb = obj.find("bndbox")
                        x1 = float(bb.find("xmin").text)
                        y1 = float(bb.find("ymin").text)
                        x2 = float(bb.find("xmax").text)
                        y2 = float(bb.find("ymax").text)
                        entry["target"]["boxes"].append([x1, y1, x2, y2])

    return formatted_predictions


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == '__main__':

    args = parse_args()
    args = set_dataset_args(args)

    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs)

    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.dataset_t, training=False)
    dataloader_t = torch.utils.data.DataLoader(
        roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, imdb_t.num_classes, training=True),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Init model
    fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN.eval()

    if args.cuda:
        fasterRCNN.cuda()

    # Input tensors
    im_data_w = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    im_info = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    gt_boxes = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    num_boxes = Variable(torch.LongTensor(1).cuda() if args.cuda else torch.LongTensor(1))

    predictions = defaultdict(list)
    data_iter = iter(dataloader_t)

    # -------------------------------------------------------------------------
    # INFERENCE LOOP
    # -------------------------------------------------------------------------

    for step in range(len(dataloader_t)):

        data = next(data_iter)
        img_path = data[-2][0]
        img_filename = os.path.basename(img_path)

        weak_data = data[0][:, 0, :, :, :]
        im_data_w.resize_(weak_data.size()).copy_(weak_data)
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, *_ = fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes)

        cls_prob = cls_prob.squeeze()
        rois = rois.squeeze()

        # Select best non-background class
        best_cls = None
        best_score = -1
        best_box = None

        for i in range(cls_prob.size(0)):
            scores = cls_prob[i]
            top_class = torch.argmax(scores).item()

            if top_class == 0:
                continue

            if scores[top_class] > best_score:
                best_score = scores[top_class].item()
                best_cls = top_class
                best_box = rois[i][1:].cpu().tolist()

        if best_cls is None:
            best_cls = 0
            best_score = 0.1
            best_box = [0, 0, 0, 0]

        predictions[best_cls].append(
            f"{img_filename} {best_score:.3f} {best_box[0]:.1f} {best_box[1]:.1f} {best_box[2]:.1f} {best_box[3]:.1f}"
        )

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    predictions = full_prediction(predictions)

    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    fpi, tpr, froc_values, fpi_levels = calculate_froc(predictions)
    plot_froc_curve(fpi_levels, froc_values)

    np.savez("froc_data.npz", fpi=fpi, tpr=tpr, froc_values=froc_values, fpi_levels=fpi_levels)

    # -------------------------------------------------------------------------
    # FIXED ASSERT
    # -------------------------------------------------------------------------
    assert len(fpi_levels) == len(froc_values), "Lengths do not match! (interpolated TPR vs FPI levels)"

    def recall_at(target_fpi):
        idx = np.argmin(np.abs(np.array(fpi_levels) - target_fpi))
        return float(froc_values[idx])

    R05 = recall_at(0.05)
    R03 = recall_at(0.30)
    R05_0 = recall_at(0.50)
    R10 = recall_at(1.00)

    auc = float(np.trapz(froc_values, fpi_levels))

    precision = np.clip(np.array(froc_values) / (np.array(fpi_levels) + 1e-6), 0, 1)
    f1_scores = 2 * precision * np.array(froc_values) / (precision + np.array(froc_values) + 1e-6)
    best_f1 = float(np.max(f1_scores))

    print("R@0.05 =", R05)
    print("R@0.3  =", R03)
    print("R@0.5  =", R05_0)
    print("R@1.0  =", R10)
    print("AUC    =", auc)
    print("Best F1 =", best_f1)

    print("done")


# -------------------------------------------------------------------------
# ===== CUSTOM R-VALUE COMPUTATION (Your Required Snippet) =====
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 300
})

# Provide your own values here
tpr_values = []
fpi_levels = []  # fill with your actual 32 FPI values

if len(tpr_values) == len(fpi_levels) and len(tpr_values) > 0:
    R05 = tpr_values[np.argmin(np.abs(np.array(fpi_levels) - 0.05))]
    R03 = tpr_values[np.argmin(np.abs(np.array(fpi_levels) - 0.3))]
    R05_0 = tpr_values[np.argmin(np.abs(np.array(fpi_levels) - 0.5))]
    R10 = tpr_values[np.argmin(np.abs(np.array(fpi_levels) - 1.0))]

    print("\nCUSTOM R VALUES:")
    print("R@0.05 =", R05)
    print("R@0.30 =", R03)
    print("R@0.50 =", R05_0)
    print("R@1.00 =", R10)

