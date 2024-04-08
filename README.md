# Joint Physical-Digital Facial Attack Detection Via Simulating Spoofing Clues [CVPR2024 Workshop]
This repository is the official implementation of [Joint Physical-Digital Facial Attack Detection Via Simulating Spoofing Clues](https://arxiv.org/)
# 🤗 Overview our method
The overview pipeline of our method. We propose Simulated Physical Spoofing Clues augmentation (SPSC), which augments live samples into simulated physical attack samples for training within protocols 1 and 2.1. Concurrently, we present Simulated Digital Spoofing Clues augmentation (SDSC), converting live samples into simulated digital attack samples, tailored for training under protocols 1 and 2.2.
<img src="readme_images/pipeline.png" alt="image" style="zoom:20%;" />


1. employ ColorJitter to simulate the spoofing clues of print attacks
2. use moire pattern augmentation to simulate the spoofing clues of replay attacks
3. SPSC consists of ColorJitter and moire pattern augmentation
4. introduce SPSC to simulate the spoofing clues of digital forgery
5. attempt to use GaussNoise or gradient-based noise to simulate the spoofing clues of adversarial attacks, but do not work

# 🏃‍♂️ Getting Started

## Data Format
Please make a copy of the original data and place it in the following format：
1. copy data and get txt files
```bash 
cp -r xxx/phase1/* cvpr2024/data/
cp -r xxx/phase2/* cvpr2024/data/
cd cvpr2024/data
cat p1/dev.txt p1/test.txt > p1/dev_test.txt
cat p2.1/dev.txt p2.1/test.txt > p2.1/dev_test.txt
cat p2.2/dev.txt p2.2/test.txt > p2.2/dev_test.txt
```
2. get train_dev_label.txt
```bash 
python data_preprocess/merge_dev_train_data.py # please modify the base path
# p1: merge all dev data to train
# p2.1: only merge dev live data to train
# p2.2: only merge dev live data to train
```
3. final data format:
```bash
base_path = "xxx/cvpr2024"

|----cvpr2024/data
  |----p1
    |----train
    |----dev
    |----test
    |----train_live_mask
    |----train_label.txt
    |----dev_label.txt
    |----train_dev_label.txt
    |----dev_test.txt
  |----p2.1
    |----train
    |----dev
    |----test
    |----train_live_mask
    |----train_label.txt
    |----dev_label.txt
    |----train_dev_label.txt
    |----dev_test.txt
  |----p2.2
    |----train
    |----dev
    |----test
    |----train_live_mask
    |----train_label.txt
    |----dev_label.txt
    |----train_dev_label.txt
    |----dev_test.txt
```

## ⚒️ Installation
```bash
pip install -r requirements.txt
```

## Data Preprocess
#### Detect face
if image width >= 700 and image height >= 700, we will detect face from the image and expand 20 pixel crop face from the image.
If no face is detected in the image, we will center crop a 500*500 bbox from the image which implements this logic in dataset.py where the training data is loaded.
we need pip install insightface and use insightface to detect face.It will download the model by default to detect faces. 
<br>
Please modify the base path and run detect_face.py.The crop_face will overwrite the original image.
```bash
cd data_preprocess
python detect_face.py
```
#### Generate train.txt live sample's mask
[face-parsing github](https://github.com/zllrunning/face-parsing.PyTorch) <br>
download face-parsing model from [79999_iter.pth](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) <br>
Please modify root_dir and model_path.
```bash
cd data_preprocess/face_parsing
bash generate_mask.sh
```
## Imagenet pretrain model
download pretrain model: [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)

## Training resources
- Please use 1*A100(80G) for training, only modify the dataset base path, do not modify other parameters in training.
- We fixed the random seed to ensure reproducible results. Modifying other training parameters will cause fluctuations in the final results.
- Training only takes 1 hour for each protocol. 
- Inference only takes 1 minute for each protocol.

## 🚀  P1 Train and Test
Train p1 protocol:
```bash
bash train_p1.sh
```
Test: select the 200th epoch model weight
```bash
bash test_p1.sh
```
## 🚀  P2.1 Train and Test
Train p2.1 protocol:
```bash
bash train_p21.sh
```
Test: select the 200th epoch model weight
```bash
bash test_p21.sh
```

## 🚀  P2.2 Train and Test
Train p2.2 protocol:
```bash
bash train_p22.sh
```
Test: select the 200th epoch model weight
```bash
bash test_p22.sh
```

