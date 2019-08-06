import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs
from utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender
from pafprocess import pafprocess
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

def find_peaks(img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > 0.1)
    out = np.zeros_like(img)
    out[peaks_binary] = img[peaks_binary]
    return out

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg
    
weight_name = './network/weight/pose_model.pth'
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            paf, heatmap = get_outputs(
                multiplier, oriImg, model,  'rtpose')
                  
    heatmap_peaks = np.zeros_like(heatmap)
    for i in range(19):
        heatmap_peaks[:,:,i] = find_peaks(heatmap[:,:,i])
    heatmap_peaks = heatmap_peaks.astype(np.float32)
    heatmap = heatmap.astype(np.float32)
    paf = paf.astype(np.float32)

    #C++ postprocessing      
    pafprocess.process_paf(heatmap_peaks, heatmap, paf)

    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)) / heatmap.shape[1],
                float(pafprocess.get_part_y(c_idx)) / heatmap.shape[0],
                pafprocess.get_part_score(c_idx)
            )

        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)
            
    out = draw_humans(oriImg, humans)

        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
