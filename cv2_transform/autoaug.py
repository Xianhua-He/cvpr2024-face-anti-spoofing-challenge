import cv2
import numpy as np
import random
from . import functional as F

random_mirror = True
INTER_MODE = cv2.INTER_LINEAR
padding_value = (0, 0, 0)


def ShearX(img, v): # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    M = np.float32([[1, v, 0],[0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=INTER_MODE, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_value)

def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    M = np.float32([[1, 0, 0],[v, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=INTER_MODE, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_value)

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert v >= 0
    if random.random() > 0.5:
        v = -v
    M = np.float32([[1, 0, v],[0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=INTER_MODE, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_value)

def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert v >= 0
    if random.random() > 0.5:
        v = -v
    M = np.float32([[1, 0, 0],[0, 1, v]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=INTER_MODE, borderMode=cv2.BORDER_CONSTANT, borderValue=padding_value)

def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return F.rotate(img, v)



def AutoContrast(img, _):
    channels = []
    cutoff = 0
    img = img.astype(np.uint8)
    for i in range(3):
        n_bins = 256
        hist = cv2.calcHist([img], [i], None, [n_bins], [0, n_bins])
        n = np.sum(hist)
        cut = cutoff * n // 100
        low = np.argwhere(np.cumsum(hist) > cut)
        low = low[0]
        high = np.argwhere(np.cumsum(hist[::-1]) > cut)
        high = n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.astype(np.uint8)
        channels.append(table[img[:, :, i]])
    out = cv2.merge(channels)
    return out

def Invert(img, _):
    return cv2.bitwise_not(img.astype(np.uint8))

def Equalize(img, _):
    return cv2.merge([cv2.equalizeHist(split) for split in cv2.split(img.astype(np.uint8))])

def Flip(img, _):
    return cv2.flip(img, 1)

def Solarize(img, v):
    assert 0 <= v <= 256
    img = img.astype(np.uint8)
    return np.where(img < v, img, 255 - img)

def SolarizeAdd(img, addition=0, threshold=128):
    img = img.astype(np.uint8)
    img = np.clip(img + addition, 0, 255)
    return np.where(img < threshold, img, 255 - img)

def Posterize(img, v): # [4, 8]
    #assert 4 <= v <= 8
    v = int(v)
    shift = 8 - v
    return np.left_shift(np.right_shift(img.astype(np.uint8), shift), shift)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return F.adjust_contrast(img.astype(np.float32), v).astype(np.uint8)#cv2.convertScaleAbs(img, alpha=v, beta=0.)

def Brightness(img, v):   # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return F.adjust_brightness(img.astype(np.float32), v).astype(np.uint8)#cv2.convertScaleAbs(img, alpha=1., beta=(v-1.)*127)

def Color(img, v):
    assert 0.1 <= v <= 1.9
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = F.blend(rgb, img, v)
    return np.clip(img, 0, 255).astype(np.uint8)

def Sharpness(img, v):
    assert 0.1 <= v <= 1.9
    img = img.astype(np.float32)
    kernel = np.array([[1,1,1],[1,5,1],[1,1,1]])
    smooth = cv2.filter2D(img, -1, kernel/13) #cv2.GaussianBlur(img, (3, 3), 13)
    img = F.blend(smooth, img, v)
    return np.clip(img, 0, 255).astype(np.uint8)

def CutoutAbs(img, v):
    # assert 0 <= v <= 20
    if v < 0:
        return img
    h, w = img.shape[:2]
    x0 = np.random.randint(w)
    y0 = np.random.randint(h)

    x0 = int(max(0, x0 - v // 2))
    y0 = int(max(0, y0 - v // 2))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    img[y0:y1, x0:x1] = 125 
    return img

def rand_augment_list():  # 16 oeprations and their ranges
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]
    return l


class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = rand_augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > random.uniform(0.2, 0.8):
                continue
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img