import os
import torch
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def is_valid_jpg(jpg_file):
    if not os.path.exists(jpg_file):
        return False
    if jpg_file.split('.')[-1].lower() in ['jpg', 'jpeg']:
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == b'\xff\xd9'
    else:
        return True
    

class ImageListDataset(Dataset):
    def __init__(self, basedir, data_list, is_train, transforms1=None, transforms2=None, return_path=False):
        self.basedir = basedir
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.return_path = return_path
        self.is_train = is_train
        fp = open(data_list, "rt")
        lines = fp.readlines()
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        fp.close()

        self.items = []
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
            while True:
                img_path, label = self.items[idx]
                image = cv2.imread(os.path.join(self.basedir, img_path))
                if image is None:
                    print('image read error, {}'.format(img_path))
                    idx = random.randrange(0, len(self.items))
                    continue
                break

            if self.transforms1 is not None:
                image = self.transforms1(image)
            if self.transforms2 is not None:
                image = self.transforms2(image=image)
                image = image['image']

            if self.return_path:
                return image, int(label), img_path
            else:
                return image, int(label)
