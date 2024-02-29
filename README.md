# MIG_Bench
The MIG benchmark of CVPR2024 MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis

### [[Paper]](https://arxiv.org/pdf/2402.05408.pdf)     [[Project Page]](https://migcproject.github.io/)
**MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis**
<br>_Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, Yi Yang_<br>
## To Do List
- [ ] Sample Code
- [ ] MIG Bench File
- [ ] Evaluation


## Installation

### Conda environment setup
```
conda create --name mig_eval python=3.8 -y
conda activate mig_eval

python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install stanza==1.4.2
pip install nltk
pip install supervision==0.7.0
pip install protobuf==3.20.2
pip install pytorch_fid
pip install imageio
```

## Evaluation Pipeline



## Evaluation Results

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
```