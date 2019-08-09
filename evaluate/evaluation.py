import unittest
import torch
from evaluate.coco_eval import run_eval
from lib.network.rtpose_vgg import get_model, use_vgg
from torch import load

#Notice, if you using the 
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    weight_name = '/home/tensorboy/Downloads/pose_model.pth'
    state_dict = torch.load(weight_name)
    model = get_model(trunk='vgg19')
    
    #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model.float()
    model = model.cuda()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    run_eval(image_dir= '/home/tensorboy/data/coco/images/val2017', anno_file = '/home/tensorboy/data/coco/annotations/person_keypoints_val2017.json', vis_dir = '/home/tensorboy/data/coco/images/vis_val2017', model=model, preprocess='rtpose')


