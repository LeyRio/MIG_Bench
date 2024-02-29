import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

import os
import json

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils
import argparse

from PIL import Image, ImageDraw, ImageFont
import groundingdino.datasets.transforms as T

from tqdm import tqdm
import torch
from transformers import CLIPProcessor,CLIPModel
from PIL import Image
from pytorch_fid import fid_score

try:  # 使用 try 机制引用 moxing 包，避免本地与云端环境频繁切换引入问题
    import moxing as mox
    is_cloud = True
except:
    is_cloud = False

from eval.ap import AveragePrecisionOnImages

import time
# 一点点检查
# 1.将目标检测结果可视化出来，看看是否与图像对齐(可视化100张)
# 2.测试100个样例，将其miou等中间数据均输出出来，查看具体效果

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../pretrained/groundingdino_swint_ogc.pth"


# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

inception_model = torchvision.models.inception_v3(pretrained=False)
state_dict = torch.load('../pretrained/inception_v3_google-0cc3c7bd.pth')
inception_model.load_state_dict(state_dict)

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "../pretrained/sam_vit_h_4b8939.pth"

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device = 'cuda')
sam_predictor = SamPredictor(sam)

clip_model = CLIPModel.from_pretrained('../pretrained/clip_tokenizer').cuda().eval()
clip_processor = CLIPProcessor.from_pretrained('../pretrained/clip_tokenizer')

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def calc_clip_score(image,prompt,need_template = False):

    prompt_list = []
    if need_template:
        for text_template in imagenet_templates:
            filled_text = text_template.format(prompt)
            prompt_list.append(filled_text)
    else:
        prompt_list.append(prompt)
    # print(prompt_list)
    inputs = clip_processor(text = prompt_list,images = image,return_tensors='pt',padding=True)
    for key in inputs.keys():
        inputs[key] = inputs[key].cuda().detach()

    outputs = clip_model(**inputs)

    torch.cuda.empty_cache()
    logits_per_image = outputs.logits_per_image

    return torch.mean(logits_per_image).cpu()



def draw_box_desc(image, gt_bbox,pred_bbox,prompt,miou):
    pred_bbox = []
    image_draw = cv2.rectangle(image,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),color=(255,0,0),thickness=5)
    if len(pred_bbox) > 0:
        image_draw_2 = cv2.rectangle(image_draw,(int(pred_bbox[0]),int(pred_bbox[1])),(int(pred_bbox[2]),int(pred_bbox[3])),color=(0,255,0),thickness=2)
    else:
        image_draw_2 = image_draw
    cv2.putText(image_draw_2,prompt,(int(gt_bbox[0] + 10),int(gt_bbox[1] + 10)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),2)
    cv2.putText(image_draw_2,str(miou),(int(gt_bbox[0] + 5),int(gt_bbox[1] + 30)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

    return image_draw_2

def check_on_image(image = None,prompt = None,gt_bbox = None,attr = None,box_t = 0.25,text_t = 0.25,miou_threshold = 0.5,args = None,image_path = None):
    # prompt 包含了 所有可能需要检测的类别 list
    countsss = 0
    segment_label = {}
    # prompt = attr
    # attr = 'red'

    attr_flag = 0
    success_flag = 0
    

    p_max = 0
    p_index = 0
    total_max = 0
    color_map = {'green color.':0,'yellow color.':1,'white color.':2,'black color.':3,'brown color.':4,'blue color.':5,'red color.':6}
    color_dict = {
        'red':[{'Lower':np.array([0,43,35]),'Upper':np.array([6,255,255])},{'Lower':np.array([156,43,35]),'Upper':np.array([180,255,255])}],
        'blue':{'Lower':np.array([78,43,35]),'Upper':np.array([124,255,255])},
        'green':{'Lower':np.array([35,43,35]),'Upper':np.array([77,255,255])},
        'yellow':{'Lower':np.array([20,43,35]),'Upper':np.array([34,255,255])},
        'black':{'Lower':np.array([0,0,0]),'Upper':np.array([180,255,35])},
        'white':{'Lower':np.array([0,0,221]),'Upper':np.array([180,43,255])},
        'brown':{'Lower':np.array([6,43,35]),'Upper':np.array([25,255,255])},
    }

    CLASSES = [prompt]  # reading from json getting classes

    BOX_THRESHOLD = box_t
    TEXT_THRESHOLD = text_t
    NMS_THRESHOLD = 0.8

    # Detecting nouns in the files

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]

        
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if detections.xyxy.shape[0] > 0:
        pred_bbox = detections.xyxy
        min_x = np.maximum(pred_bbox[:,0],gt_bbox[0])
        max_x = np.minimum(pred_bbox[:,2],gt_bbox[2])
        min_y = np.maximum(pred_bbox[:,1],gt_bbox[1])
        max_y = np.minimum(pred_bbox[:,3],gt_bbox[3])
        iw = np.maximum(max_x - min_x, 0.)
        ih = np.maximum(max_y - min_y, 0.)
        insert_area = iw * ih
        union_area = (pred_bbox[:,2] - pred_bbox[:,0]) * (pred_bbox[:,3] - pred_bbox[:,1]) + (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]) - insert_area
        iou = insert_area / union_area
        ovmax = np.max(iou) # 最大重叠
        if ovmax < miou_threshold:
            return 0,0,ovmax
        else:
            success_flag = 1
            miou = ovmax
    else:
        return 0,0,0.0
        

    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # print(masks)
            index = np.argmax(scores)
            maskk = np.asfortranarray(masks[index])
            maskk = mask_utils.encode(maskk)
            maskk['counts'] = maskk['counts'].decode('utf-8')
            result_masks.append(maskk)
        return result_masks

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=gt_bbox[None,:]
    )
    
    mask_obj = mask_utils.decode(detections.mask)  


    color_dic = color_dict[attr]
    if args.debug:
        print(f'checking target color is {attr}')

    # 根据物体mask去除背景，替代为灰色背景
    segment_mask = torch.from_numpy(mask_obj)
        
    detect_mask = torch.zeros(size=(512,512))

    for mask_id in range(segment_mask.shape[2]):
        mask = segment_mask[:,:,mask_id]
        detect_mask = torch.logical_or(mask,detect_mask).int()

    image_mid = image * (detect_mask.unsqueeze(-1).detach().numpy())

    rev_mask = (1 - detect_mask).unsqueeze(-1).detach().numpy()
    color_bg = np.zeros([512,512,3]).astype(np.uint8) + 127
    image_mid = image_mid + rev_mask * color_bg
    image_mid = image_mid.astype(np.uint8)



    color_mask = check_on_color_cv(image_mid,prompt,color_dic,attr,args,image_path)

    color_mask = torch.from_numpy(color_mask)
    final_mask = torch.logical_and(detect_mask,color_mask).int()
    # 根据颜色占比 与 实际物体占比 判断物体颜色
    # print(torch.sum(final_mask)/torch.sum(detect_mask))
    if torch.sum(detect_mask) == 0.0 or torch.sum(final_mask)/torch.sum(detect_mask) < 0.2:
        if args.debug:
            if torch.sum(detect_mask) == 0.0:
                print(f'we can not fine the object {prompt}')
            else:
                print(f'the color of {prompt} is wrong')
        attr_flag = 0
        miou = 0.0
    else:
        if args.debug:
            print(f'color of {prompt} is right')
        attr_flag = 1

    return success_flag, attr_flag, miou

def check_on_color_cv(image = None,class_name = None,color_dict = None,color_type = None,args = None,image_path = None):
    dist_image = np.array(image.shape,image.dtype)
    dist_image = cv2.cvtColor(image,code=cv2.COLOR_BGR2HSV,dst = dist_image)
    if isinstance(color_dict,list):
        mask = np.zeros([512,512],np.uint8)
        for color_dic in color_dict:
            lower = color_dic['Lower']
            upper = color_dic['Upper']

            result_mask = cv2.inRange(dist_image,lower,upper)/255
            mask = np.logical_or(result_mask,mask).astype(np.int_)
        mask = mask * 255
    else:
        lower = color_dict['Lower']
        upper = color_dict['Upper']
        mask = cv2.inRange(dist_image,lower,upper)
    result_mask = mask

    image_final = image * result_mask[:,:,np.newaxis]
    image_final = image_final.astype(np.uint8)
    output_dir = './output/'
    image_final = cv2.cvtColor(image_final,code=cv2.COLOR_HSV2BGR)
    if args.debug:
        cv2.imwrite(os.path.join(output_dir, f"{color_type}_{class_name}_pred{image_path}.jpg"),result_mask[:,:,np.newaxis])
        cv2.imwrite(os.path.join(output_dir, f"{color_type}_{class_name}_origin{image_path}.jpg"),image)

    return result_mask

# 命名标题时，采用 blue cat_red dog_42.jpg的形式，物体的顺序与出现的位置严格对应
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="path to image file",default='obs://bucket-1664-huadong/code/ley1233020/A_Eval_Image/Fuser/fuser_output/attr_bench/eval_exp55_e161/model_ley_config_21/')
    parser.add_argument('--gt_dir',type=str,help='path to origin image',default=None)
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--need_clip_score',action='store_true')
    parser.add_argument('--need_sucess_ratio',action='store_true')
    parser.add_argument('--need_visor_score',action='store_true')
    parser.add_argument('--need_local_clip',action='store_true')
    parser.add_argument('--need_miou_score',action='store_true')
    parser.add_argument('--need_instance_visor_cond',action='store_true')
    parser.add_argument('--metric_name',type=str,default='test')
    parser.add_argument('--miou_threshold',type=float,default=0.5)
    parser.add_argument('--need_instance_sucess_ratio',action='store_true')
    parser.add_argument('--save_obs_path',type=str,default='obs://bucket-veddata01/code/ley1233020/A_Metric/')
    parser.add_argument('--debug_file_path',type=str,default='./debug_path/')
    
    args = parser.parse_args()
    args.need_clip_score = True
    args.need_local_clip = True
    args.need_sucess_ratio = True
    args.need_visor_score = True
    args.need_miou_score = True
    args.need_instance_visor_cond = True
    args.need_instance_sucess_ratio = True
    args.debug = False
    miou_threshold = args.miou_threshold

    

    # 加载参数和标注真实文件
    image_dir = args.image_dir
    gt_dir = args.gt_dir

    # 如果需要的话，从云端将图像copy过来
    ori_image_dir = image_dir
    if not os.path.exists(image_dir):
        if is_cloud:
            print('copying generated image')
            if os.path.isdir('/cache/test_sample/'):
                rm_dir = '/cache/test_sample/'
                os.system(f'rm -rf {rm_dir}')
            mox.file.copy_parallel(image_dir,'/cache/test_sample/')
            image_dir = '/cache/test_sample/'
        else:
            print('There is no picture!!!!')
            args.need_clip_score = False
            args.need_local_clip = False
            args.need_sucess_ratio = False
            args.need_visor_score = False
    image_path_list = os.listdir(image_dir)


    coco_path = './sample_prompt_color_split.json'
    with open(coco_path,'r') as coco_file:
        coco_context = json.load(coco_file)

    if args.debug:
        if os.path.isdir(args.debug_file_path):
            os.system(f'rm -rf {args.debug_file_path}')
        os.makedirs(args.debug_file_path, exist_ok=True)



    count = 0 
    need_check_instance = args.need_sucess_ratio or args.need_local_clip or args.need_visor_score or args.need_instance_sucess_ratio or args.need_miou_score or args.need_instance_visor_cond
    need_segment_instance = args.need_sucess_ratio or args.need_visor_score or args.need_instance_sucess_ratio or args.need_miou_score or args.need_instance_visor_cond
    need_crop_instance =  args.need_local_clip

    clip_record = 0.0
    clip_count = 0
    
    loca_clip_record = 0.0
    loca_clip_count = 0
    
    miou_record = 0.0
    miou_count = 0
    miou_level_record = [0.0,0.0,0.0,0.0,0.0]
    miou_level_count = [0,0,0,0,0]


    sucess_record = 0.0
    sucess_count = 0
    success_level_record = [0,0,0,0,0]
    sucess_level_count = [0,0,0,0,0]


    inst_suceess_count = 0
    inst_count = 0
    inst_success_level_count = [0,0,0,0,0]
    inst_level_count = [0,0,0,0,0]

    miou_instance_level = [0.0,0.0,0.0,0.0,0.0]
    miou_instance_count = [0,0,0,0,0]

    visor_instance_count = 0
    visor_acc_both_count = 0
    visor_acc_obj_only_count = 0
    visor_acc_obj_count = 0

    visor_level_instance_count = [0,0,0,0,0]
    visor_level_acc_both_count = [0,0,0,0,0]
    visor_level_acc_obj_only_count = [0,0,0,0,0]
    visor_level_acc_obj_count = [0,0,0,0,0]


    inst_cond_count = 0
    inst_cond_success = 0
    inst_level_cond_count = [0,0,0,0,0]
    inst_level_cond_success = [0,0,0,0,0]


    visor_acc_obj_dict = {}
    visor_acc_att_dict = {}
    visor_acc_both_dict = {}
    visor_level_acc_obj_dict = [{},{},{},{},{}]
    visor_level_acc_att_dict = [{},{},{},{},{}]
    visor_level_acc_both_dict = [{},{},{},{},{}]


    metric_result = {}

    gt_per_image = {}
    counts = 0
    gt_per_image = {}
    gt_count = {}

    # 添加iou和单物体成功率
    for image_path in tqdm(image_path_list):
        # 获取这张图片对应的image_id
        image_id = image_path.split('_')[-1].split('.')[0]
        coco_info = coco_context[image_id]
        counts = counts + 1

        caption = coco_info['caption']
        gt_bbox_list = coco_info['segment']

        # 读取图像
        image_abs_path = os.path.join(image_dir,image_path)
        image = cv2.imread(image_abs_path)
        if image.shape[0] != 512:
            image = cv2.resize(image, dsize=(512, 512))
        image_draw = image.copy()
        level = len(gt_bbox_list) - 2
        # image_pil,image = load_image(image_abs_path)
        # while True:
        # if counts > 10:
        #     break
        instance_count = 0
        instance_s_count = 0


        if args.need_clip_score:

            clip_score = calc_clip_score(image,caption)
            clip_record = clip_record + clip_score.item()
            clip_count = clip_count + 1

        if need_check_instance:

            # 先统计当前图像所有物体 的 类别，并存储其对应的真实值物体框
            if image_id not in visor_acc_obj_dict.keys():
                visor_acc_obj_dict[image_id] = []
                visor_acc_att_dict[image_id] = []
                visor_acc_both_dict[image_id] = []
                visor_level_acc_obj_dict[level][image_id] = []
                visor_level_acc_att_dict[level][image_id] = []
                visor_level_acc_both_dict[level][image_id] = []

            if need_segment_instance:
                sucess_obj_per_image = 1
                sucess_attr_per_image = 1

                for gt_instance in gt_bbox_list:
                    instance_count = instance_count + 1
                    label_w_attr = gt_instance['label']
                    label = " ".join(label_w_attr.split(" ")[1:])
                    attr = label_w_attr.split(" ")[0]
                    gt_bbox = np.array(gt_instance['bbox']) * 512
                    

                    if args.need_visor_score or args.need_sucess_ratio or args.need_instance_sucess_ratio or args.need_miou_score or args.need_instance_visor_cond:
                        sucess_obj,sucess_attr,miou = check_on_image(image,label,gt_bbox,attr,miou_threshold = miou_threshold,args = args,image_path = image_path)
                        sucess_obj_per_image = sucess_obj_per_image * sucess_obj
                        sucess_attr_per_image = sucess_attr_per_image * sucess_attr

                    if sucess_obj:
                        instance_s_count = instance_s_count + 1

                    if args.need_miou_score:
                        miou_record = miou_record + miou
                        miou_count = miou_count + 1
                        miou_level_record[level] = miou_level_record[level] + miou
                        miou_level_count[level] = miou_level_count[level] + 1

                    if args.need_instance_sucess_ratio:
                        inst_count = inst_count + 1
                        inst_level_count[level] = inst_level_count[level] + 1
                        if sucess_obj and sucess_attr:
                            inst_suceess_count = inst_suceess_count + 1
                            inst_success_level_count[level] = inst_success_level_count[level] + 1

                    if args.need_instance_visor_cond:
                        if sucess_obj:
                            inst_level_cond_count[level] = inst_level_cond_count[level] + 1
                            inst_cond_count = inst_cond_count + 1

                            if sucess_attr:
                                inst_cond_success = inst_cond_success + 1
                                inst_level_cond_success[level] = inst_level_cond_success[level] + 1
                        

                if args.need_sucess_ratio:
                    if sucess_obj_per_image * sucess_attr_per_image == 1:
                        sucess_record = sucess_record + 1
                        success_level_record[level] = success_level_record[level] + 1
                    sucess_count = sucess_count + 1
                    sucess_level_count[level] = sucess_level_count[level] + 1

                if args.need_visor_score:
                    visor_acc_obj_dict[image_id].append(sucess_obj_per_image)
                    visor_acc_att_dict[image_id].append(sucess_attr_per_image)
                    visor_acc_both_dict[image_id].append(sucess_obj_per_image * sucess_attr_per_image)

                    visor_level_acc_obj_dict[level][image_id].append(sucess_obj_per_image)
                    visor_level_acc_att_dict[level][image_id].append(sucess_attr_per_image)
                    visor_level_acc_both_dict[level][image_id].append(sucess_obj_per_image * sucess_attr_per_image)
                    
                    visor_instance_count = visor_instance_count + 1
                    visor_level_instance_count[level] =  visor_level_instance_count[level] + 1  
                    if sucess_obj_per_image:
                        visor_acc_obj_only_count = visor_acc_obj_only_count + 1
                        visor_level_acc_obj_only_count[level] = visor_level_acc_obj_only_count[level] + 1

                    if sucess_obj_per_image and sucess_attr_per_image:
                        visor_acc_both_count = visor_acc_both_count + 1
                        visor_level_acc_both_count[level] = visor_level_acc_both_count[level] + 1
                    
                    
            if need_crop_instance:  

                for instance in gt_bbox_list:
                    inst_bbox = instance['bbox']
                    inst_label = instance['label']
                    cropped_image = image[int(512 * inst_bbox[1]):int(512 * inst_bbox[3]),int(512 * inst_bbox[0]):int(512 * inst_bbox[2]),:]
                    cropped_image = cv2.resize(cropped_image,(512,512))

                    if args.need_local_clip:
                        # 是否需要补足成为模板？
                        local_clip_score = calc_clip_score(cropped_image,inst_label,need_template = True)
                        loca_clip_record = loca_clip_record + local_clip_score.item()
                        loca_clip_count = loca_clip_count + 1

    
    # 输出并保存结果
    print(f'Here is the metric:')
    if args.need_clip_score:
        clip_score = clip_record / clip_count
        metric_result['clip_score'] = clip_score
        print(f'CLIP score : {clip_score}')
    if args.need_local_clip:
        local_clip_score = loca_clip_record / loca_clip_count
        metric_result['local_clip_score'] = local_clip_score
        print(f'Local CLIP: {local_clip_score}')

    if args.need_sucess_ratio:
        sucess_ratio = sucess_record / sucess_count
        sucess_level_ratio = [0.0,0.0,0.0,0.0,0.0]
        for i in range(5):
            sucess_level_ratio[i] = success_level_record[i] / sucess_level_count[i]
        metric_result['sucess_ratio'] = sucess_ratio
        metric_result['success_level_ratio'] = sucess_level_ratio
        print(f'SUCESS RATIO: {sucess_ratio}')
        print(f'SUCESS LEVEL RATIO: {sucess_level_ratio}')
    
    if args.need_visor_score:
        visor_level_score = [0.0,0.0,0.0,0.0,0.0]
        visor_level_cond_score = [0.0,0.0,0.0,0.0,0.0]
        visor_level_num = [[0 for i in range(8)] for j in range(5)]

        visor_score = visor_acc_both_count/visor_instance_count
        visor_cond_score = visor_acc_both_count/ (visor_acc_obj_only_count + 1e-5)
        # 逐句统计 VISOR1/2/3/4/5/6/7/8
        visor_num = [0 for i in range(8)]
        for p_promt in visor_acc_obj_dict.keys():
            for i in range(8):
                if sum(visor_acc_both_dict[p_promt]) >= (i + 1):
                    visor_num[i] = visor_num[i] + 1
        for i in range(8):
            visor_num[i] = visor_num[i] / (len(list(visor_acc_obj_dict.keys())) + 1e-5)


        for i in range(5):
            visor_level_score[i] = visor_level_acc_both_count[i] / visor_level_instance_count[i]
            visor_level_cond_score[i] = visor_level_acc_both_count[i] / (visor_level_acc_obj_only_count[i] + 1e-5)
            for p_promt in visor_level_acc_obj_dict[i].keys():
                for j in range(8):
                    if sum(visor_level_acc_obj_dict[i][p_promt]) >= (j + 1):
                        visor_level_num[i][j] = visor_level_num[i][j] + 1
            for j in range(8):
                visor_level_num[i][j] = visor_level_num[i][j] / (len(list(visor_level_acc_obj_dict[i].keys())) + 1e-5)

        metric_result['visor_score'] = visor_score
        metric_result['visor_cond_score'] = visor_cond_score
        metric_result['visor_num_score'] = visor_num
        metric_result['visor_level_score'] = visor_level_score
        metric_result['visor_level_cond_score'] = visor_level_cond_score
        metric_result['visor_level_num_score'] = visor_level_num
        metric_result['Instance_split_rate'] = instance_s_count / instance_count

        print(f'VISOR SCORE: {visor_score}')
        print(f'VISOR_COND SCORE: {visor_cond_score}')
        print(f'VISOR_num SCORE: {visor_num}')

        print(f'VISOR LEVEL SCORE: {visor_level_score}')
        print(f'VISOR_LEVEL_COND SCORE: {visor_level_cond_score}')
        print(f'VISOR_LEVEL_num SCORE: {visor_level_num}')
        print(f'Instance_split_rate: {instance_s_count / instance_count}')

    if args.need_instance_sucess_ratio:
        inst_level_sr = [0.0,0.0,0.0,0.0,0.0]
        inst_sr = inst_suceess_count / inst_count
        for i in range(5):
            inst_level_sr[i] = inst_success_level_count[i] / inst_level_count[i]
        metric_result['inst_sucess_ratio'] = inst_sr
        metric_result['inst_level_sucess_ratio'] = inst_level_sr
        print(f'INST SUCESS RATIO: {inst_sr}')
        print(f'INST Level SUCESS RATIO: {inst_level_sr}')

    if args.need_miou_score:
        miou_level_score = [0.0,0.0,0.0,0.0,0.0]
        miou_score = miou_record / miou_count
        for i in range(5):
            miou_level_score[i] = miou_level_record[i] / miou_level_count[i]
        metric_result['miou'] = miou_score
        metric_result['miou_level'] = miou_level_score
        print(f'MIOU SCORE: {miou_score}')
        print(f'MIOU LEVEL SCORE : {miou_level_score}')

    if args.need_instance_visor_cond:
        instance_level_visor_cond = [0.0,0.0,0.0,0.0,0.0]
        instance_visor_cond = inst_cond_success / inst_cond_count
        for i in range(5):
            instance_level_visor_cond[i] = inst_level_cond_success[i] / inst_level_cond_count[i]
        metric_result['instance_visor_cond'] = instance_visor_cond
        metric_result['instance_level_visor_cond'] = instance_level_visor_cond
        print(f'INST VISOR COND: {instance_visor_cond}')
        print(f'INST LEVEL VISOR COND: {instance_level_visor_cond}')
        


    metric_result['metric_name'] = args.metric_name
    metric_result['image_path'] = ori_image_dir
    
    
    if is_cloud:
        result = json.dumps(metric_result)
        with open(f'./metric_{args.metric_name}.json','w') as output_f:
            output_f.write(result)
        # mox.file.copy_parallel(f'./metric_{args.metric_name}.json',os.path.join(args.save_obs_path,f'./metric_{args.metric_name}.json'))
        print('Evaluation is Over!!!')
        

        

        


            



