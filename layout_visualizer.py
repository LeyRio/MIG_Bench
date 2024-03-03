# code for visualize your layout
import cv2
import numpy as np


import os
import json
import torch

import argparse

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from PIL import Image

used_dict = torch.zeros((512,512))

import time
import shutil

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"


def draw_box_desc_v2(image, gt_bbox,color_type,prompt,mode = 'bbox',text_bg = True):

    color_dict = {'yellow':(0,255,255),'red':(0,0,255),'green':(0,255,0),'blue':(255,0,0),'black':(0,0,0),'white':(255,255,255),'brown':(42,42,165)}
    color = color_dict[color_type]
    
    bar_h = 35
    text_len = 23
    
    image_draw2 = image

    candidate_y_list = [(int(gt_bbox[1]),1),(int(gt_bbox[3]),-1),(int(gt_bbox[1]),-1),(int(gt_bbox[3]),1)]
        
    size,bottom = cv2.getTextSize(prompt,cv2.FONT_HERSHEY_TRIPLEX, 0.75, 1)
    width, height = size

    x1 = int(gt_bbox[0])
    y1 = int(gt_bbox[1])
    x2 = int(gt_bbox[0] + width)
    y2 = int(gt_bbox[1] + height + bottom)

    flag = 1
    now_state = 0
    if x2 >= 512:
        x2 = 511
        x1 = 511 - width
    # adjust y
    while torch.sum(used_dict[x1:x2,y1:y2] > 0) or (y2 >= 510) or (y1 <= 0):
        # change annotate position
        now_state = now_state + 1
        if now_state >= len(candidate_y_list):
            break
        y1,h = candidate_y_list[now_state]
        y2 = y1 + h * (height + bottom)
        flag = h

    used_dict[x1:x2,y1:y2] = 1
        
    if mode == 'bbox':
        image_draw2 = cv2.rectangle(image_draw2,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),color=color,thickness=5)
        if text_bg:
            image_draw2 = cv2.rectangle(image_draw2,(x1,y1),(x2,y2),color=color,thickness=-1)
    else:
        
        if color_type in ['yellow','green','white']:
            type_color = color_dict['black']
        else:
            type_color = color_dict['white']

        pos_y1 = y1 + flag * (height + bottom)//2 

        cv2.putText(image_draw2,prompt,(x1,pos_y1),cv2.FONT_HERSHEY_TRIPLEX,0.75,type_color,1)
        # size,bottom = cv2.getTextSize('Fmg_b',cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        # import pdb;pdb.set_trace()
    # cv2.putText(image_draw_2,str(miou),(int(gt_bbox[0] + 5),int(gt_bbox[1] + 30)),cv2.FONT_HERSHEY_COMPLEX,0.5,colors,2)

    return image_draw2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="path to image file",default='/data1/liyou/code/GLIGEN/generation_samples/generation_box_text')
    parser.add_argument('--output_name',type=str,default='fuser')
    parser.add_argument('--file_path',type=str,default='/data1/liyou/code/MIG/mig_bench.json')
    parser.add_argument('--output_dir',type=str,default='./annotation/',help = 'path to output annotation')
    
    '''
    Note that all image naming formats in the image_dir must end with _cocoid 
    (for example: xxx_{seed}_{coco_id}.png)
    in order to find the corresponding layout information in the JSON file through this tool.
    '''

    args = parser.parse_args()

    image_dir = args.image_dir
    ori_image_dir = image_dir
    output_name = args.output_name
    file_path = args.file_path
    output_dir = args.output_dir

    image_path_list = os.listdir(image_dir)

    # read layout yaml file
    with open(file_path,'r') as coco_file:
        coco_context = json.load(coco_file)

    # create annotation file path
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir,output_name)
    if os.path.isdir(output_path):
        os.system(f'rm -rf {output_path}')
    os.makedirs(output_path,exist_ok=True)
    
    for image_path in tqdm(image_path_list):
        image_id = image_path.split('_')[-1].split('.')[0]

        save_number = 0
        if len(image_path.split('_')) > 2:
            save_number = image_path.split('_')[1]

        coco_info = coco_context[image_id]
        caption = coco_info['caption']
        gt_bbox_list = coco_info['segment']

        # image_abs_path = os.path.join(image_dir,image_path)
        # Creating gray background
        img = np.zeros((512,512,3),np.uint8)
        img.fill(200)
        image_draw = img
        index = 0

        used_dict[:,:] = 0
        for gt_bbox in gt_bbox_list:
            inst_bbox = [i* 512 for i in gt_bbox['bbox']]
            labels = gt_bbox['label']
            color = labels.split(' ')[1]
            image_draw = draw_box_desc_v2(image_draw,inst_bbox,color,labels,mode = 'bbox')

        used_dict[:,:] = 0
        for gt_bbox in gt_bbox_list:
            inst_bbox = [i* 512 for i in gt_bbox['bbox']]
            labels = gt_bbox['label']
            color = labels.split(' ')[1]
            image_draw = draw_box_desc_v2(image_draw,inst_bbox,color,labels,mode = 'text')

        cv2.imwrite(os.path.join(f'./annotation/{output_name}',f'{image_path}_{save_number}_{image_id}_annotation.jpg'),image_draw)
                    