import json
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path',type=str,default='/data/coco/annotations/captions_val2014.json')
    parser.add_argument('--segment_path',type=str,default='/data/coco/annotations/instances_val2014.json')
    parser.add_argument('--file_name',type=str,default='coco_mig.json')
    args = parser.parse_args()
    
    caption_path = args.caption_path
    segment_path = args.segment_path

    with open(caption_path,'r') as caption_file:
        caption = json.load(caption_file)

    with open(segment_path,'r') as segment_file:
        segment = json.load(segment_file)


    image_id_list = []

    caption_list = caption['annotations'] 
    segment_list = segment['annotations'] 
    image_info_list = caption['images']
    categories = segment['categories']

    category_dict = {}
    for cate in categories:
        category_dict[cate['id']] = cate['name']

    final_dict = {}

    image_info_dict = {}
    segment_dict = {}

    for image_info in tqdm(image_info_list):

        image_id = image_info['id']
        image_info_dict[image_id] = image_info

    for segment in tqdm(segment_list):
        image_id = segment['image_id']
        if image_id in segment_dict.keys():
            segment_dict[int(image_id)].append(segment)
        else:
            segment_dict[int(image_id)] = [segment]

    for cap_context in tqdm(caption_list):
        
        image_id = cap_context['image_id']
        if image_id not in image_info_dict.keys():
            # print(image_id)
            continue

        instance_dict = {}
        instance_dict['caption'] = cap_context['caption']
        has_instance = True
        
        if image_id not in final_dict.keys():
            final_dict[image_id] = {}
            matching_image_info = image_info_dict[image_id]
            H = matching_image_info['height']
            W = matching_image_info['width']
            final_dict[image_id]['h'] = H
            final_dict[image_id]['w'] = W
            has_instance = False
        else:
            H = final_dict[image_id]['h']
            W = final_dict[image_id]['w']
            has_instance = True
        if image_id in segment_dict.keys():
            matching_segment = segment_dict[image_id]
            instance_list = []
        else:
            instance_list = []
            matching_segment = []
        
        for instance in matching_segment:
            in_dict = {}
            bbox = instance['bbox']
            x,y,w,h = bbox
            x2 = x + w
            y2 = y + h
            if W!=0 and H!=0:
                bbox_final = [x/W,y/H,x2/W,y2/H]
            in_dict['bbox'] = bbox_final
            instance_id = instance['category_id']
            in_dict['label'] = category_dict[int(instance_id)]
            instance_list.append(in_dict)

        instance_dict['segment'] = instance_list
        if not has_instance:
            final_dict[image_id]['instance'] = [instance_dict]
        else:
            final_dict[image_id]['instance'].append(instance_dict)

    json_str = json.dumps(final_dict)
    with open(args.file_name, 'w') as json_file:
        json_file.write(json_str)
