# Resample MIG-Bench

## Step 1. Downloading COCO dataset
First, download the coco2014val or coco2014test file from [COCO dataset](https://cocodataset.org/#download) or Google Drive [COCO](https://drive.google.com/file/d/1fdQUhwRLAhRAW0XJccki0WC_t9OCBrhD/view?usp=sharing)

Unzip the annotation files and put it under ./annotations/ foler.

(For example, the path of captions_val2014.json is ./annotations/captions_val2014.json)

## Step 2. Organize information
Next, integrate all the layout information and the corresponding labels:
```
python mig_bench_prepare.py \
--caption_path ./annotations/captions_val2014.json \
--segment_path ./annotations/instances_val2014.json
```
Or you can download the coco_mig.json on Google Drive [URL](https://drive.google.com/file/d/1kmz0aYs_N9A05F7UpKeLh0nb3aiksFrw/view?usp=drive_link) and place it in the project's main folder.

## Step 3. Resample
Finally, run the following code to resample the new bench:
```
python mig_bench_sample.py --file_path ./coco_mig.json --file_name 'mig_bench.json'
```
The file_path parameter is the file path generated in Step 2. You can adjust many other parameters in the code.
