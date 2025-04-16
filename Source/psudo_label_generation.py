# coding:utf-8
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import xml.etree.ElementTree as ET
import xml.dom.minidom
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss


from model.utils.parser_func import parse_args, set_dataset_args
from model.ema.optim_weight_ema import WeightEMA
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.roi_layers import nms
# from model.utils.loss import Entropy, score_function
# from model.utils.obtain_predictions import obtain_predictions

from PIL import Image
import matplotlib.pyplot as plt

# ===Rajes===
import logging

# Setup logging to save logs to step2.log
logging.basicConfig(
    filename='logs/pseudo_labels.log', 
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import torch
import numpy as np
import cv2


import xml.etree.ElementTree as ET
import xml.dom.minidom

def save_voc_xml(output_path, img_filename, img_path, im_info, rois_mean, g_mean, l_mean, class_mapping, confidence_threshold=0):
    # Read original image to get actual dimensions
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    
    real_height, real_width = img.shape[:2]

    # im_info gives size used during network forward
    net_height = float(im_info[0][0].item())
    net_width = float(im_info[0][1].item())

    # Compute scaling ratios
    scale_x = real_width / net_width
    scale_y = real_height / net_height

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(img_path))
    ET.SubElement(annotation, "filename").text = img_filename
    ET.SubElement(annotation, "path").text = img_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    ET.SubElement(source, "annotation").text = "VOC"
    ET.SubElement(source, "image").text = "flickr"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(real_width)
    ET.SubElement(size, "height").text = str(real_height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    for i in range(g_mean.size(0)):
        top2_classes = torch.topk(g_mean[i], 2) 
        max_class = top2_classes.indices[0].item()
        class_confidence = top2_classes.values[0].item()

        if max_class not in class_mapping:
            continue
        if max_class == 0 or class_confidence < confidence_threshold:
            continue

        start_idx = max_class * 4
        if start_idx + 4 <= l_mean.size(1):
            bbox = l_mean[i, start_idx:start_idx + 4]

        # Scale bounding box from resized to original dimensions
        xmin = int(rois_mean[i][1] * scale_x)
        ymin = int(rois_mean[i][2] * scale_y)
        xmax = int(rois_mean[i][3] * scale_x)
        ymax = int(rois_mean[i][4] * scale_y)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_mapping[max_class]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "score").text = str(float(class_confidence))

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xml_str = ET.tostring(annotation, encoding="utf-8")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml_str)





if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    logging.info(f'Called with args:')
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    logging.info(cfg)
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset

    # ===Rajes===
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target, training=False)
    train_size_t = len(roidb_t)
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = f"{args.save_dir}/{args.net}/{args.log_ckpt_name}/Annotations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb_t.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data_s = torch.FloatTensor(1)
    im_data_w = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data_s = im_data_s.cuda()
        im_data_w = im_data_w.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data_s = Variable(im_data_s)
    im_data_w = Variable(im_data_w)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    # from model.faster_rcnn.vgg16_adv import vgg16
    from model.faster_rcnn.vgg16_adv import vgg16
    # from model.faster_rcnn.resnet_HTCN import resnet

    if args.net == 'vgg16':
        # fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_t.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc, la_attention = args.LA_ATT, mid_attention = args.MID_ATT)

    else:
        print("network is not defined")
        # pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load pretrain checkpoint %s" % (args.load_name)) #--load_name
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load pretrain model successfully!')

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()


    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)


    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)


    count_iter = 0
    img_paths = []
    img_paths_similar = []
    img_paths_disimilar = []
    A_score_list = []
    H_list = []
    img_paths_no_label = []
    data_iter_t = iter(dataloader_t)
    zero_count = 0

    fasterRCNN.eval()
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    fasterRCNN.apply(apply_dropout)
    logging.info(f"Total Step: {len(dataloader_t)}")


# Define class mapping

# for medical dataset
# class_mapping = {
#     0: "__background__",
#     1: "mal"
# }

# for natural dataset
class_mapping = {
    0: "__background__",
    1: 'bus',
    2:'bicycle', 
    3:'car', 
    4:'motorcycle', 
    5:'person', 
    6:'rider', 
    7:'train', 
    8:'truck'
}


for step in range(len(dataloader_t)):
    data_t = next(data_iter_t)
    img_path = data_t[-2][0]
    img_filename = os.path.basename(img_path)
    xml_path = os.path.join(output_dir, img_filename.replace(".jpg", ".xml"))
    if os.path.exists(xml_path):
        print(f"Skipping {img_filename} as {xml_path} already exists.")
        continue
    weak_aug_data = data_t[0][:, 0, :, :, :]
    im_data_w.resize_(weak_aug_data.size()).copy_(weak_aug_data)
    im_info.resize_(data_t[1].size()).copy_(data_t[1])
    gt_boxes.resize_(1, 1, 5).zero_()
    num_boxes.resize_(1).zero_()
    

    T = 1
    for t in range(T):
        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, out_d_pixel, out_d = fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes)
        if t == 0:
            g = cls_prob
            l = bbox_pred
            rois_c=rois
        g = torch.cat((g, cls_prob), dim=0)
        l = torch.cat((l, bbox_pred), dim=0)
        rois_c = torch.cat((rois_c, rois), dim=0)
        del rois, cls_prob, bbox_pred
        torch.cuda.empty_cache() 
    
    g_mean = torch.mean(g, dim=0)
    l_mean = torch.mean(l, dim=0)
    rois_mean = torch.mean(rois_c, dim=0)
    g_mean = g_mean.squeeze(0)
    l_mean = l_mean.squeeze(0)
    rois_mean = rois_mean.squeeze(0)
    
    # print(f"g_mean[]: {g_mean.size()} - {g_mean[0]}")
    # print(f"g_mean[]: {g_mean.size()} - {g_mean[0]}")
    logging.info(f"Image name: {img_filename}")
    width, height, depth = int(im_info[0][1]), int(im_info[0][0]), 3  # Adjust as needed
    
    

    save_voc_xml(
        output_path=xml_path,
        img_filename=img_filename,
        img_path=img_path,
        im_info=im_info,
        rois_mean=rois_mean,
        g_mean=g_mean,
        l_mean=l_mean,
        class_mapping=class_mapping,
        confidence_threshold=0.8
    )


    # logging.info("VOC XML file saved to {xml_path}")
    del data_t
    del g,l,rois_c,g_mean,l_mean,rois_mean
    torch.cuda.empty_cache()

print('done')
logging.info(f"done")



