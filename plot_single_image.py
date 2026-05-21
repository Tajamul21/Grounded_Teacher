import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from GT.modeling.meta_arch.rcnn import (
    TwoStagePseudoLabGeneralizedRCNN,
    DAobjTwoStagePseudoLabGeneralizedRCNN
)

from GT.modeling.proposal_generator.rpn import PseudoLabRPN
from GT.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from GT.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from GT.modeling.meta_arch.vgg import build_vgg_backbone  # noqa

from GT import add_cat_config


def setup_cfg(config_file, model_path, score_thresh=0.9):
    cfg = get_cfg()
    add_cat_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # adjust if needed
    cfg.freeze()
    return cfg


def run_inference(image_path, cfg, save_path=None):
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    print(f"Total detections (score > {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}): {len(instances)}")

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) > 0 else None

    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.0
    )

    vis_output = v.draw_instance_predictions(instances)

    result_img = vis_output.get_image()

    # Show
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, result_img[:, :, ::-1])
        print(f"Saved result to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single image inference with bbox visualization")
    parser.add_argument("--config", default="/DATA/Tawheed/SFDA/Grounded_Teacher/download/ddsm2inbreast/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="/DATA/Tawheed/SFDA/Grounded_Teacher/download/ddsm2inbreast/model_0029999.pth", help="Path to model .pth")
    parser.add_argument("--image", default="/DATA/Tawheed/SFDA/Grounded_Teacher/download/ddsm2inbreast/images/bcd/image2.png", help="Path to input image")
    parser.add_argument("--score", default=0.3, type=float, help="Confidence threshold")
    parser.add_argument("--output", default="plots/result.png", help="Save path")

    args = parser.parse_args()

    cfg = setup_cfg(
        config_file=args.config,
        model_path=args.checkpoint,
        score_thresh=args.score
    )

    run_inference(args.image, cfg, args.output)
