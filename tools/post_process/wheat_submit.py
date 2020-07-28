import json
import os
import argparse
import cv2
import torch

from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test

import matplotlib.pyplot as plt


import sys
sys.path.insert(0, "./weighted-boxes-fusion")
from ensemble_boxes import *
import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    config = 'configs/wheat/htc_dconv_c3-c5_mstrain_600_1000_x101_64x4d_fpn_12e_wheat.py'
    checkpoint = '../output/gwd/htc_dcn_rext101/epoch_9.pth'

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

    results = []
    fig, ax = plt.subplots(5, 2, figsize=(30, 70))
    count = 0

    for data_info, result in zip(dataset.data_infos, outputs):
        img_path = cfg.data.test['img_prefix'] + '/' + data_info['filename']
        img = cv2.imread(img_path)

        # boxes, scores, labels = run_wbf(result[0])

        boxes = result[0][:, :4]
        scores = result[0][:, 4]


        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >=float(0.05)]

        #show save result
        if count<10:
            for box, score in zip(boxes,scores):
                cv2.rectangle(img,
                              (box[0], box[1]),
                              (box[2]+box[0], box[3]+box[1]),
                              (220, 0, 0), 2)
                cv2.putText(img, '%.2f'%(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite('submit/gwd/img_'+ str(count) + '.jpg', img) 
            ax[count%5][count//5].imshow(img)
            count+=1

        result = {
            'image_id': data_info['filename'][:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)

    
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

    # save result
    test_df.to_csv('submission.csv', index=False)
    test_df.head()



