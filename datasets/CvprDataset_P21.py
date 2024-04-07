import os
import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from data_augmentation.moire import Moire
from torchvision import transforms
from PIL import Image


class CvprDataset_P21(Dataset):
    def __init__(self, basedir, data_list, transforms1=None, transforms2=None, is_train=True, return_path=False):
        self.base_path = basedir
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.return_path = return_path
        self.is_train = is_train
        self.items = []

        # data_augmentation
        self.moire = Moire()

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

        if mode == 'train' and label == 0 and self.is_train:
            prob_value = random.random()
        #
        #     # physical
            if prob_value < 0.3:
                label = 1
                image = self.moire(image)

            elif prob_value >= 0.3 and prob_value < 0.6:
                label = 1
                color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                transformed_image_pil = color_jitter(image_pil)
                transformed_image_np = np.array(transformed_image_pil)
                image = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)

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