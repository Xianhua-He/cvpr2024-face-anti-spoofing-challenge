from __future__ import absolute_import

import random
import math



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.0, 0.0, 0.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) <= self.probability:
            height, width = img.shape[1:]
            area = width * height
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                    
            h = min(int(round(math.sqrt(target_area * aspect_ratio))), height)
            w = min(int(round(math.sqrt(target_area / aspect_ratio))), width)
            y1 = random.randint(0, height - h)
            x1 = random.randint(0, width - w)
            img[0, y1:y1+h, y1:y1+w] = self.mean[0]
            img[1, y1:y1+h, y1:y1+w] = self.mean[1]
            img[2, y1:y1+h, y1:y1+w] = self.mean[2]

        return img