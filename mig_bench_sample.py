import json
from tqdm import tqdm
from random import choice
import cv2
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type=str,default='coco_mig.json',help = 'the path of the output file of step 1')
    parser.add_argument('--sample_num',type=int,default=800)
    parser.add_argument('--remove_person',type=bool,default = True)
    parser.add_argument('--file_name',type=str,default='mig_bench.json')
    args = parser.parse_args()

    file_path = args.file_path
    sample_num = args.sample_num

    color_list = ['red','blue','green','yellow','black','white','brown']

    with open(file_path,'r') as json_file:
        json_f = json.load(json_file)

    key_list = list(json_f.keys())
    used_dict = []
    choose_dict = {}
    inst_count = [0,0,0,0,0]
    count = 0

    name_list = []
    
    while True:
        key = choice(key_list)
        if key not in used_dict:
            inst = json_f[key]['instance']
            caption_inst = choice(inst)
            prompt = caption_inst['caption']
            objects = caption_inst['segment']
            segment_list = []
            
            for obj in objects:
                if (obj['bbox'][2] - obj['bbox'][0]) < 1/8  or (obj['bbox'][3] - obj['bbox'][1]) < 1/8 :
                    continue
                segment_list.append(obj)

            save = True
            if len(segment_list) > 1:
                new_seg_list = []
                overlap_list = []
                if len(segment_list) > 6:
                    new_caption_inst = {}
                    area_list = []
                    for obj in segment_list:
                        area = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                        area_list.append(area)
                    combined_list = list(zip(area_list[:6],segment_list[:6]))
                    sorted_combined_list = sorted(combined_list,key=lambda x:x[0],reverse=True)
                    _, sorted_segment = zip(*sorted_combined_list)
                    p_ = 'a photo of '
                    cp_list = []
                    for seg in sorted_segment:
                        color = choice(color_list)
                        if seg['label'] == 'person' and args.remove_person:
                            save = False
                        obj_label = 'a ' + color + ' ' + seg['label']
                        cp_list.append(obj_label + ' ')
                        new_seg_list.append({'bbox':seg['bbox'],'label':obj_label})
                    p_ = p_ + 'and '.join(cp_list)
                else:
                    new_caption_inst = {}
                    p_ = 'a photo of '
                    cp_list = []
                    for seg in segment_list:
                        color = choice(color_list)
                        if seg['label'] == 'person' and args.remove_person:
                            save = False
                        obj_label = 'a ' + color + ' ' + seg['label']
                        cp_list.append(obj_label + ' ')
                        new_seg_list.append({'bbox':seg['bbox'],'label':obj_label})
                    p_ = p_ + 'and '.join(cp_list)
                if len(new_seg_list) >=2 and inst_count[len(new_seg_list) - 2] < sample_num // 5 and p_ not in name_list and save:
                    name_list.append(p_)
                    new_caption_inst['caption'] = p_
                    new_caption_inst['segment'] = new_seg_list
                    new_caption_inst['image_id'] = key
                    choose_dict[count] = new_caption_inst
                    inst_count[len(new_seg_list) - 2] = inst_count[len(new_seg_list) - 2] + 1
                    count = count + 1
                used_dict.append(key)
        else:
            continue
        if count == sample_num:
            break

    json_str = json.dumps(choose_dict)
    with open(args.file_name, 'w') as json_file:
        json_file.write(json_str)
    print('Sample Over!!!')
