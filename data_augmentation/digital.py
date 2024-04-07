from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import random
import albumentations as A
from albumentations import Compose, ChannelShuffle, RGBShift, HueSaturationValue, RandomBrightnessContrast,Downscale, Sharpen,LongestMaxSize,Resize,PadIfNeeded,CenterCrop, Affine
import joblib
import elasticdeform
import tqdm


def digital_augment(srcFace, mask_path, loc='all', mode='minor'):

    src_img, tar_img, spatial_transform = st_generator(srcFace, mode)

    # read mask
    raw_hullMask = convex_hull(mask_path, loc)    # size (h, w, c) mask of face convex hull
    # mask deformation
    hullMask = spatial_transform(image=raw_hullMask)["image"]

    elt_mask = elasticdeform.deform_random_grid(hullMask, sigma=4, points=6)

    kernal_id = random.sample([0,1],2)
    kernal_sets = [(3,3),(5,5)]
    blur1 = cv2.GaussianBlur(elt_mask, kernal_sets[kernal_id[0]], 3)
    blured = cv2.GaussianBlur(blur1, kernal_sets[kernal_id[1]], 3)

    r = random.sample([0.25, 0.5, 0.75, 1, 1, 1], 1) if mode == 'minor' else random.sample([0.75, 1, 1, 1], 1)
    forge_mask = r*blured
    forge_mask = np.expand_dims(forge_mask, -1).repeat(3, axis=-1)

    bld_img = forge(src_img, tar_img, forge_mask)
    return bld_img
    

def transform_params(img, mode='minor'):

    if(mode=='minor'):
        color_transform=A.Compose([
            # ChannelShuffle(p=0.1),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=10, always_apply=False, p=0.5),
            # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
            Downscale(scale_min=0.35, scale_max=0.65, interpolation=0, always_apply=False, p=0.4),
            Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            ])
    else:
        color_transform=A.Compose([
            ChannelShuffle(p=0.03),
            HueSaturationValue(hue_shift_limit=9, sat_shift_limit=9, val_shift_limit=20, always_apply=False, p=0.5),
            # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.65, interpolation=0, always_apply=False, p=0.4),
            Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            ])

    h, w = img.shape[:2]

    if (mode=='minor'):
        ampr, ampt = 0.02,0.02
    else:
        ampr, ampt = 0.04,0.04

    size = h
    umin, umax = 1-ampr, 1+ampr
    uh,uw = random.uniform(umin, umax), random.uniform(umin, umax)
    hr, wr = int(uh*h), int(uw*w)   # resize

    vmin, vmax =  -ampt, ampt
    vh, vw = random.uniform(vmin, vmax), random.uniform(vmin, vmax)
    th, tw = int(vh*h), int(vw*w)   # translate

    spatial_transform=A.Compose([
        Resize(hr,wr),
        # LongestMaxSize(max_size=512, interpolation=1, always_apply=False, p=1),
        PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT,value=0),
        CenterCrop(height=h, width=w, always_apply=False, p=1.0),
        Affine(translate_px={'x':tw, 'y':th},  cval=0, mode=0, p=0.5)
    ])

    change_params = [hr, wr, th, tw]
    return color_transform, spatial_transform, change_params


def st_generator(src_img, mode):
    tar_img = src_img.copy()
    color_transform, spatial_transform, change_params = transform_params(src_img, mode)

    select_prob = random.random()
    flag = 0 if (select_prob < 0.5) else 1   # color change selection : "0-sr 1-tar"
    if(flag == 0):
        src_img = color_transform(image=src_img)["image"]
    else:
        tar_img = color_transform(image=tar_img)["image"]
    
    src_img = spatial_transform(image=src_img)["image"]

    return src_img, tar_img, spatial_transform


def forge(src_img, tar_img, mask):
    return (mask * src_img + (1 - mask) * tar_img).astype(np.uint8)


def convex_hull(mask_path, loc):

    atts = {1:'skin', 2:'l_brow', 3:'r_brow', 4:'l_eye', 5:'r_eye', 6:'eye_g', 7:'l_ear', 8:'r_ear', 9:'ear_r',
            10:'nose', 11:'mouth', 12:'u_lip', 13:'l_lip', 14:'neck', 15:'neck_l', 16:'cloth', 17:'hair', 18:'hat'}

    mask = joblib.load(mask_path)
    mask = mask.astype(np.uint8)

    if(loc=='all'):
        where_res = np.where((mask>=1) & (mask<=6) | (mask>=10) & (mask<=13)) # 1-6 10-13
    elif(loc=='skin'):
        where_res = np.where(mask==1) # 1
    elif(loc=='top'):
        where_res = np.where((mask>=2) & (mask<=6)) # 2-6
    elif(loc=='nose'):
        where_res = np.where(mask==10) # 1
    elif(loc=='mouth'):
        where_res = np.where((mask>=11) & (mask<=13)) # 11-13

    
    bin_mask = np.zeros_like(mask, dtype='float32')
    bin_mask[where_res] = 1

    return bin_mask


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*
    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped


if __name__ == '__main__':
    image = cv2.imread('005700.jpg')
    mask_path = "005700.pkl"
    digital_augment(image, mask_path, loc='all', mode='minor')
