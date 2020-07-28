import json
import os
import argparse
import cv2
import torch

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numba
import re
import ast
import xml.etree.ElementTree as ET
from typing import List, Union, Tuple

from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test

import matplotlib.pyplot as plt


import sys
sys.path.insert(0, "/home/chen/ai-competition/weighted-boxes-fusion")
from ensemble_boxes import *





#pascal_voc: min/max coordinates [x_min, y_min, x_max, y_max]
#coco: width/height instead of maxes [x_min, y_min, width, height]
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):
        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)
        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1
        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)

def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


def show_result(sample_id, pred_boxes, pred_scores, gt_boxes,save_name):
    image_path = f'/home/chen/data/global-wheat-detection/test/{sample_id}.jpg'
    sample = cv2.imread(image_path, cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for pred_box,score in zip(pred_boxes,pred_scores):
        cv2.rectangle(
            sample,
            (pred_box[0], pred_box[1]),
            (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]),
            (220, 0, 0), 2
        )
        cv2.putText(sample, '%.2f'%(score), (pred_box[0], pred_box[1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (255,255,255), 2, cv2.LINE_AA)


    for gt_box in gt_boxes:    
        cv2.rectangle(
            sample,
            (gt_box[0], gt_box[1]),
            (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]),
            (0, 0, 220), 2
        )

    ax.set_axis_off()
    ax.imshow(sample)
    # fig.show(sample)
    ax.set_title("RED: Predicted | BLUE - Ground-truth")
    fig.savefig(save_name)

def run_wbf(prediction, image_size=1024, iou_thr=0.43, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[:, :4]/(image_size-1)).tolist()]
    scores = [(prediction[:,4]).tolist()]
    labels = [(np.ones(prediction[:,4].shape[0])).tolist() ]

    boxes, scores, labels = nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)
    boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def parse_voc_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        # obj_struct['bbox'] = [xmin,ymin,w,h]
        objects.append([xmin,ymin,w,h])

    return objects

def parse_args():
    parser = argparse.ArgumentParser(description='global wheat detection submit')
    parser.add_argument('--config', help='config python file path', default = '../output/gwd/htc_dcn_rext101/htc_dconv_c3-c5_mstrain_600_1000_x101_64x4d_fpn_12e_wheat.py', type=str)
    parser.add_argument('--checkpoint', help='weights file path', default = '../output/gwd/htc_dcn_rext101/epoch_9.pth', type=str)
    # parser.add_argument('--submit-file-name', help='submit_file_name', default = './submit/gwd/submission.csv', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    config = args.config
    checkpoint = args.checkpoint

    # test_img = '/home/chen/ai-competition/global_wheat_detection/test/51b3e36ab.jpg'
    # model = init_detector(config, checkpoint, device='cuda:0')

    # result = inference_detector(model, test_img)
    # show_result_pyplot(model,test_img,result,score_thr=0.3)

    cfg = Config.fromfile(config)
    cfg.data.test.test_mode = True
    distributed = False

    #set device cpu or gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #build dataset
    dataset = build_dataset(cfg.data.test)
    print(cfg.data.test)

    test_img_prefix = cfg.data.test['img_prefix']
    test_label_prefix = cfg.data.test['img_prefix'].replace('test','test_voc_label')
    print(test_label_prefix)

    #build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False)

    #build detector
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    #load weights
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu') # 'cuda:0'

    model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, False)

    # results = []
    # fig, ax = plt.subplots(5, 2, figsize=(30, 70))
    # count = 0
    ##WBF
    wbf_iou_thr = 0.5
    wbf_skip_box_thr = 0.3

    test_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.75, 0.05)]
    for index, (data_info, result) in enumerate(zip(dataset.data_infos, outputs)):
        image_id = data_info['filename'].split('.')[0]
        img_path = test_img_prefix + '/' + data_info['filename']
        xml_path = test_label_prefix + '/' + data_info['filename'].replace('.jpg','.xml')
        # print(img_path)
        # print(xml_path)
        img = cv2.imread(img_path)
        gt_boxes = parse_voc_xml(xml_path) #xmin,ymin,w,h

        boxes, scores, labels = run_wbf(result[0],image_size=1024, iou_thr=wbf_iou_thr, skip_box_thr=wbf_skip_box_thr)

        # boxes = result[0][:, :4]
        # scores = result[0][:, 4]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >=float(0.05)]

        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = boxes[preds_sorted_idx]

        save_name = str(index) + '.jpg'
        show_result(image_id, preds_sorted, scores, gt_boxes,save_name)

        image_precision = calculate_image_precision(preds_sorted,
                                                    gt_boxes,
                                                    thresholds=iou_thresholds,
                                                    form='coco')
        print(image_precision)
        
        test_image_precisions.append(image_precision)

    print("test image precisions mAP50(0.5:0.95): {0:.4f}".format(np.mean(test_image_precisions)))



