import os
import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from data_augmentation.digital import digital_augment


class CvprDataset_P22(Dataset):
    def __init__(self, basedir, data_list, transforms1=None, transforms2=None, is_train=True, return_path=False):
        self.base_path = basedir
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.return_path = return_path
        self.is_train = is_train
        self.items = []

        # center crop
        self.bbox_delta = 250

        with open(data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if is_train:
                    items = line.strip().split()
                    img_path = items[0]
                    label = items[1]
                    self.items.append((img_path, label))
                else:
                    items = line.strip().split()
                    img_path = items[0]
                    if len(items) == 1:
                        label = "1"
                    else:
                        label = items[1]
                    self.items.append((img_path, label))

    def __getitem__(self, idx):
        mode = ""
        while True:
            fpath = self.items[idx][0]
            if 'train' in fpath:
                mode = 'train'
            elif 'dev' in fpath:
                mode = 'dev'
            elif 'test' in fpath:
                mode = 'test'

            image = cv2.imread(os.path.join(self.base_path, fpath))

            height, width, _ = image.shape
            if height >= 700 and width >= 700:
                center_x = width // 2
                center_y = height // 2
                image = image[center_y-self.bbox_delta: center_y+self.bbox_delta, center_x-self.bbox_delta:center_x+self.bbox_delta, :]

            if image is None:
                print('image read error, {}'.format(fpath))
                idx = random.randrange(0, len(self.items))
                continue
            break

        label = int(self.items[idx][1])

        if label == 0 and mode == 'train' and self.is_train:
            prob_value = random.random()

        #     # # digital
            if prob_value < 0.5:
                label = 1
                protocol = fpath.split('/')[0]
                img_name = fpath.split('/')[-1].split('.')[0]
                mask_path = os.path.join(self.base_path, protocol, "train_live_mask", f"{img_name}.pkl")
                bld_img = digital_augment(image, mask_path)
                image = bld_img

        if self.transforms1 is not None:
            image = self.transforms1(image)

        if self.transforms2 is not None:
            image = self.transforms2(image=image)
            image = image['image']

        if self.return_path:
            return image, label, fpath
        else:
            return image, label

    def __len__(self):
        return len(self.items)