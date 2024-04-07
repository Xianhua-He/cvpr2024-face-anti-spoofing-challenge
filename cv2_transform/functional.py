import torch
import cv2
import numpy as np
import numbers
import random
import math
from collections.abc import Sequence, Iterable
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def imread(img_name, mode=0):
    """Read image to ``numpy.ndarray``.
    Args:
        img_name (string): Image path.
        mode (string): 0 means ``RGB``, 1 means ``BGR``, default=``RGB``.
    Returns:
        numpy.ndarray: Converted image.
    """
    try:
        img = cv2.imread(img_name)
        if mode == 0:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    except:
        img = Image.open(img_name)
        if mode == 0:
            img = img.convert("RGB")
        else:
            img = img.convert("BGR")
        img = np.asarray(img)
    return img


def blend(img1, img2, alpha):
    """Blend two ``numpy.ndarray`` to a new ``numpy.ndarray``.
    Args:
        img1 (numpy.ndarray): Image one.
        img2 (numpy.ndarray): Image two.
        alpha (float): blend factor.
    Returns:
        Tensor: Converted image.
    """
    if alpha == 0:
        return img1
    elif alpha == 1:
        return img2
    else:
        return img1 * (1.0 - alpha) + img2 * alpha


def to_tensor(pic, norm_value, nchw):
    """Convert a ``numpy.ndarray`` to tensor.
    Args:
        pic (numpy.ndarray): Image to be converted to tensor.
        norm_value (int): Divide value for input.
        nchw (bool): Data format is nchw or nhwc.
    Returns:
        Tensor: Converted image.
    """
    if not _is_numpy_image(pic):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]
        if nchw:
            pic = pic.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(pic)
        return img.float().div(norm_value)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def resize(img, size, interpolation=cv2.INTER_LINEAR, shorter=True):
    r"""Resize the input numpy.ndarray to the given size.
    Args:
        img (numpy.ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR`
        shorter (bool, optional): Resize shorter edge to size.
    Returns:
        numpy.ndarray: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if shorter:
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
        else:
            if w < h:
                oh = size
                ow = int(size * w / h)
            else:
                ow = size
                oh = int(size * h / w)
        return cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, dsize=(size[1], size[0]), interpolation=interpolation)


def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given numpy.ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        numpy.ndarray: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if padding_mode == 'constant':
        # RGB image
        if len(img.shape) == 3:
            if isinstance(fill, numbers.Number):
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode, constant_values=fill)
            elif isinstance(fill, tuple):
                img_split = np.split(img, 3, 2)
                img_split_0 = np.pad(img_split[0], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode, constant_values=fill[0])
                img_split_1 = np.pad(img_split[1], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode, constant_values=fill[1])
                img_split_2 = np.pad(img_split[2], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode, constant_values=fill[2])
                img = np.concatenate((img_split_0, img_split_1, img_split_2), axis=2)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode, constant_values=fill)
    else:
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

    return img


def crop(img, top, left, height, width):
    """Crop the given numpy.ndarray.
    Args:
        img (numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        numpy.ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    return img[top:top+height, left:left+width]


def center_crop(img, output_size):
    """Crop the given numpy.ndarray and resize it to desired size.
    Args:
        img (numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        numpy.ndarray: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = output_size
    crop_top = int((image_height - crop_height) // 2)
    crop_left = int((image_width - crop_width) // 2)
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img, output_size):
    """Crop the given numpy.ndarray and resize it to desired size.
    Args:
        img (numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        numpy.ndarray: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = output_size
    crop_top = int((image_height - crop_height) // 2)
    crop_left = int((image_width - crop_width) // 2)
    img_center = crop(img, crop_top, crop_left, crop_height, crop_width)
    img_top_left = crop(img, 0, 0, crop_height, crop_width)
    img_top_right = crop(img, 0, image_width-crop_width, crop_height, crop_width)
    img_down_left = crop(img, image_height-crop_height, 0, crop_height, crop_width)
    img_down_right = crop(img, image_height-crop_height, image_width-crop_width, crop_height, crop_width)
    return (img_center, img_top_left, img_top_right, img_down_left, img_down_right)


def resized_crop(img, top, left, height, width, size, interpolation=cv2.INTER_LINEAR):
    """Crop the given numpy.ndarray and resize it to desired size.
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
    Args:
        img (numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``.
    Returns:
        numpy.ndarray: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be numpy.ndarray'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given numpy.ndarray.
    Args:
        img (numpy.ndarray): Image to be flipped.
    Returns:
        numpy.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    return img[:, ::-1]


def vflip(img):
    """Vertically flip the given numpy.ndarray.
    Args:
        img (numpy.ndarray): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    return img[::-1, :]


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy.ndarray): numpy.ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    new_img = np.zeros_like(img)
    img = blend(new_img, img, brightness_factor)

    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (numpy.ndarray): numpy.ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy.ndarray: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    mean = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.uint8).mean() + 0.5
    new_img = np.zeros_like(img) + mean
    img = blend(new_img, img, contrast_factor)

    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy.ndarray): numpy.ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy.ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    img = blend(new_img, img, saturation_factor)

    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy.ndarray): numpy.ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy.ndarray: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    new_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)

    # uint8 addition take cares of rotation across boundaries
    new_img[..., 0] = new_img[..., 0] + hue_factor * 180
    img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (numpy.ndarray): Image to be converted to grayscale.
    Returns:
        numpy.ndarray: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.uint8)
    elif num_output_channels == 3:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def rotate(img, angle, resample=cv2.INTER_LINEAR, expand=False, center=None, fill=0):
    """Rotate the image by angle.
    Args:
        img (numpy.ndarray): numpy.ndarray to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    h, w = img.shape[:2]
    if center == None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1)
    if expand:
        cos_angle = math.cos(math.pi * angle / 180)
        sin_angle = math.sin(math.pi * angle / 180)
        dsize = (int(w * sin_angle + w * cos_angle), int(h * sin_angle + h * cos_angle))
    else:
        dsize = (w, h)
    rotated = cv2.warpAffine(img, M, dsize, flags=resample, borderValue=fill)

    return rotated


def fastrotate(img, mode=1):
    """Rotate the image by angle.
    Args:
        img (numpy.ndarray): numpy.ndarray to be rotated.
        angle (int): In degrees degrees counter clockwise order.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    rotated = cv2.rotate(img, mode)

    return rotated
