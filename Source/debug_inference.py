# coding:utf-8
# python eval_debug.py --dataset_t voc_2007_test --load_name "/DATA/Tawheed/SFDA/Grounded_Teacher/Source/output/vgg16/brats/lg_adv_session_1_epoch_6_step_10000.pth"

from __future__ import absolute_import, division, print_function
import json, os, numpy as np, pprint, xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn

import _init_paths
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.parser_func import parse_args, set_dataset_args
from model.faster_rcnn.vgg16_adv import vgg16

import logging
logging.basicConfig(
    filename='logs/eval_map_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
#                 IoU + AP + mAP FUNCTIONS
# ============================================================================

def compute_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter

    return inter / union


def voc_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.0
    return ap


def full_prediction(predictions, score_threshold=0.001):
    ann_path = "/DATA/Tawheed/Dataset/Natural/VOCFoggy/VOC2007/Annotations_full/"
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


def compute_map(predictions, iou_threshold=0.5):
    per_class_ap = {}
    gt_by_image = {p["file_name"]: p["target"]["boxes"] for p in predictions}
    class_preds = defaultdict(list)

    for p in predictions:
        file = p["file_name"]
        for b, s, c in zip(p["pred"]["boxes"], p["pred"]["scores"], p["pred"]["cls_pred"]):
            class_preds[c].append((file, s, b))

    for cls, cls_predictions in class_preds.items():

        preds = sorted(cls_predictions, key=lambda x: -x[1])
        tp, fp = [], []
        gt_used = {}
        total_gt = 0

        for p in predictions:
            total_gt += len(p["target"]["boxes"])

        for file, score, box in preds:
            gt_boxes = gt_by_image[file]
            best_iou = 0
            best_idx = -1

            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(box, gt)
                best_iou = max(best_iou, iou)
                if iou == best_iou:
                    best_idx = j

            if best_iou >= iou_threshold:
                if file not in gt_used:
                    gt_used[file] = []
                if best_idx not in gt_used[file]:
                    tp.append(1)
                    fp.append(0)
                    gt_used[file].append(best_idx)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(total_gt + 1e-6)
        prec = tp / np.maximum(tp + fp, 1e-6)
        ap = voc_ap(rec, prec)

        per_class_ap[cls] = ap

    mAP = np.mean(list(per_class_ap.values())) if len(per_class_ap) else 0.0
    return mAP, per_class_ap

# ============================================================================
#                               MAIN
# ============================================================================

if __name__ == '__main__':

    args = parse_args()
    args = set_dataset_args(args)

    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs)

    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(
        args.dataset_t, training=False)

    dataloader_t = torch.utils.data.DataLoader(
        roibatchLoader(roidb_t, ratio_list_t, ratio_index_t,
                       args.batch_size, imdb_t.num_classes, training=True),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN.eval()

    if args.cuda:
        fasterRCNN.cuda()

    # Inputs
    im_data_w = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    im_info = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    gt_boxes = Variable(torch.FloatTensor(1).cuda() if args.cuda else torch.FloatTensor(1))
    num_boxes = Variable(torch.LongTensor(1).cuda() if args.cuda else torch.LongTensor(1))

    predictions = defaultdict(list)
    data_iter = iter(dataloader_t)

    print("\n========== DEBUG MODE: PROCESSING FIRST 5 IMAGES ==========\n")
    debug_counter = 0

    # ============================================================================
    #                           DEBUG INFERENCE LOOP
    # ============================================================================

    for step in range(len(dataloader_t)):

        if debug_counter >= 5:
            break

        data = next(data_iter)
        img_path = data[-2][0]
        img_filename = os.path.basename(img_path)

        print("\n------------------------------------------------------")
        print(f"[IMAGE {debug_counter+1}] {img_filename}")
        print("------------------------------------------------------")

        # Load weak image
        weak_data = data[0][:, 0, :, :, :]
        im_data_w.resize_(weak_data.size()).copy_(weak_data)
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

        rois, cls_prob, bbox_pred, *_ = fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes)

        cls_prob = cls_prob.squeeze()
        rois = rois.squeeze()

        # Print ROIs
        print("\nROI predictions:")
        for i in range(cls_prob.size(0)):
            scores = cls_prob[i]
            top_class = torch.argmax(scores).item()
            print(f" ROI[{i}] → class={top_class}, score={scores[top_class].item():.4f}, box={rois[i][1:].tolist()}")

        # Pick best prediction
        best_cls, best_score, best_box = None, -1, None

        for i in range(cls_prob.size(0)):
            scores = cls_prob[i]
            top_class = torch.argmax(scores).item()
            if top_class == 0:  # background
                continue
            if scores[top_class] > best_score:
                best_score = scores[top_class].item()
                best_cls = top_class
                best_box = rois[i][1:].cpu().tolist()

        if best_cls is None:
            print("\n⚠ No valid class prediction (all background).")
            best_cls, best_score, best_box = 0, 0, [0, 0, 0, 0]

        print(f"\nBEST prediction → class={best_cls}, score={best_score:.4f}, box={best_box}")

        # ===============================================================
        #                       GT BOX LOADING
        # ===============================================================
        ann_dir = "/DATA/Tawheed/Dataset/Natural/VOCFoggy/VOC2007/Annotations_full/"
        xml_file = os.path.join(ann_dir, img_filename.replace(".jpg", ".xml"))

        print(f"\nLooking for annotation → {xml_file}")

        if not os.path.exists(xml_file):
            xml_file = os.path.join(ann_dir, img_filename.replace(".png", ".xml"))
            if not os.path.exists(xml_file):
                print("❌ XML not found!")
                debug_counter += 1
                continue

        root = ET.parse(xml_file).getroot()
        gt_list = []

        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
            gt_list.append([x1, y1, x2, y2])

        print("\nGround truth boxes:")
        if len(gt_list) == 0:
            print("❌ No GT boxes found.")
        else:
            for b in gt_list:
                print(" GT →", b)

        # ------------------------- IoU CHECK -------------------------
        print("\nIoU comparison:")
        for gt in gt_list:
            iou = compute_iou(best_box, gt)
            print(f" IoU(pred, GT) = {iou:.4f}")
            if iou >= 0.5:
                print("  → ✅ TP (IoU >= 0.5)")
            else:
                print("  → ❌ FP (IoU < 0.5)")

        # Store prediction
        predictions[best_cls].append(
            f"{img_filename} {best_score:.3f} {best_box[0]:.1f} "
            f"{best_box[1]:.1f} {best_box[2]:.1f} {best_box[3]:.1f}"
        )

        debug_counter += 1

    print("\n========== DEBUG COMPLETE ==========\n")
