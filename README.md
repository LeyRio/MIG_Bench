# MIG_Bench
The MIG benchmark of CVPR2024 MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis

### [[Paper]](https://arxiv.org/pdf/2402.05408.pdf)     [[Project Page]](https://migcproject.github.io/)  [[Code]](https://github.com/limuloo/MIGC) 
**MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis**
## To Do List
- [x] MIG Bench File
- [x] Evaluation
- [x] Sample Code
- [ ] More baselines

## Introduction

In the Text-to-Image task, facing complex texts with multiple instances and rich attributes along with their layout information, higher demands are placed on existing generators and their derived generation techniques. In order to evaluate the generation capability of these techniques on complex instances and attributes, we designed the COCO-MIG benchmark.


The MIG bench is based on COCO images and their layouts, using the color attribute of instances as the starting point. It filters out layouts with smaller areas and instances related to humans and assigns a random color to each instance. Through specific templates, it can also construct a global prompt for each image. This bench, constructed in this way, not only retains the relatively natural distribution of COCO but also introduces complex attributes and counterfactual cases through random color assignment, greatly increasing the difficulty of generation, thus making it challenging.

During evaluation, we utilize the GroundedSAM model to detect and segment each instance. We then analyze the distribution of colors in the HSV color space for each object and calculate the proportion of the corresponding color to determine if the object's color meets the requirements. By calculating the proportion of instances correctly generated in terms of attributes and positions, along with their MIOU, we reflect the model's performance in position and attribute control.

You can find more details in our [Paper](https://arxiv.org/pdf/2402.05408.pdf).

## Installation

### Conda environment setup
```
conda create --name eval_mig python=3.8 -y
conda activate eval_mig

conda config --append channels conda-forge
conda install pytorch==1.11.0 torchvision cudatoolkit=11.3.1 -c pytorch

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/

python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

pip install opencv-python pycocotools matplotlib onnxruntime onnx nltk imageio supervision==0.7.0 protobuf==3.20.2 pytorch_fid
```

Note that you should install GroundingDINO on the GPU in order to properly run the evaluation code with cuda. If you encounter problems, you can refer to [Issue](https://github.com/IDEA-Research/GroundingDINO/issues/175) for more details.

### Checkpoints
To run the evaluation process, you need to download some model weights.

Download the GroundingDINO checkpoint:
```
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
You shoule also download ViT-H SAM model in [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

You can also manually download the weights for [Bert](https://huggingface.co/google-bert/bert-base-uncased/tree/main).

If you want to test CLIP scores, you'll also need to download the [CLIP](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) model weights.


Put all these checkpoints under ../pretrained/ folder:
```
├── pretrained
│   ├── bert-base-uncased
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── clip
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.json
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
```

## Evaluation Pipeline

### Step 1 (Optional) Sampling MIG prompts
You can choose to resample prompts for evaluation. You can check [the entire steps of Resampling](./bench_resample.md).

You can also generate your image on the 800 prompts that have been sampled from [MIG-Bench](https://drive.google.com/drive/folders/1mXxO7miVqgTq3N6q2QS7gFp_ML-qpsw2?usp=sharing).

### Step 2 Generation
Use the sampled prompts and layouts to generate images.

You can try our [MIGC](https://github.com/limuloo/MIGC) method, hope you enjoy it.

### Step 3 Evaluation
Finally, you can start evaluating your model now.

```
python eval_mig.py \
--need_miou_score \
--need_instance_sucess_ratio \
--metric_name 'eval' \
--image_dir /path/of/image/
```

## Evaluation Results
We re-sampled a version of the COCO-MIG benchmark, filtering out examples related to humans. Based on the new version of bench, we sampled 800 images and compared them with InstanceDiffusion, GLIGEN, etc. On MIG-Bench, the results are shown below. You can also find the image results and bench layout information that we generate in some of the methods in the [Example](https://drive.google.com/drive/folders/1UyhNpZ099OTPy5ILho2cmWkiOH2j-FrB?usp=sharing).



<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Method</th>
      <th colspan="6" style="text-align: center;">MIOU↑</th>
      <th colspan="6" style="text-align: center;">Instance Success Rate↑</th>
	  <th rowspan="2" style="text-align: center;">Model Type</th>
    <th rowspan="2" style="text-align: center;">Publication</th>
    </tr>
	<tr>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
	  <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
	<tr>
      <td><a href="https://github.com/showlab/BoxDiff">Box-Diffusion</a></td>
      <td>0.37</td>
      <td>0.33</td>
      <td>0.25</td>
      <td>0.23</td>
      <td>0.23</td>
      <td>0.26</td>
	  <td>0.28</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>0.12</td>
      <td>0.13</td>
      <td>0.16</td>
	  <td>Training-free</td>
    <td>ICCV2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/gligen/GLIGEN">Gligen</a></td>
      <td>0.37</td>
      <td>0.29</td>
      <td>0.253</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>0.27</td>
	<td>0.42</td>
      <td>0.32</td>
      <td>0.27</td>
      <td>0.27</td>
      <td>0.28</td>
      <td>0.30</td>
	  <td>Adapter</td>
    <td>CVPR2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/microsoft/ReCo">ReCo</a></td>
      <td>0.55</td>
      <td>0.48</td>
      <td>0.49</td>
      <td>0.47</td>
      <td>0.49</td>
      <td>0.49</td>
	  <td>0.63</td>
      <td>0.53</td>
      <td>0.55</td>
      <td>0.52</td>
      <td>0.55</td>
      <td>0.55</td>
	  <td>Full model tuning</td>
    <td>CVPR2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/frank-xwang/InstanceDiffusion">InstanceDiffusion</a></td>
      <td>0.52</td>
      <td>0.48</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.42</td>
      <td>0.46</td>
	  <td>0.58</td>
      <td>0.52</td>
      <td>0.55</td>
      <td>0.47</td>
      <td>0.47</td>
      <td>0.51</td>
	  <td>Adapter</td>
    <td>CVPR2024</td>
    </tr>
	<tr>
      <td><a href="https://github.com/limuloo/MIGC">Ours</a></td>
      <td><b>0.64</b></td>
      <td><b>0.58</b></td>
      <td><b>0.57</b></td>
      <td><b>0.54</b></td>
      <td><b>0.57</b></td>
      <td><b>0.56</b></td>
	  <td><b>0.74</b></td>
      <td><b>0.67</b></td>
      <td><b>0.67</b></td>
      <td><b>0.63</b></td>
      <td><b>0.66</b></td>
      <td><b>0.66</b></td>
	  <td>Adapter</td>
    <td>CVPR2024</td>
    </tr>
  </tbody>
</table>

## Acknowledgements
MIG-Bench is based on [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [SAM](https://github.com/facebookresearch/segment-anything), and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). We appreciate their outstanding contributions.



## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```
@misc{zhou2024migc,
      title={MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis}, 
      author={Dewei Zhou and You Li and Fan Ma and Xiaoting Zhang and Yi Yang},
      year={2024},
      eprint={2402.05408},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```