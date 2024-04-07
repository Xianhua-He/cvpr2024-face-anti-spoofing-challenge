import torch
import cv2
import numpy as np
import numbers
import random
import warnings
import math
from collections.abc import Sequence, Iterable
from . import functional as F


_cv2_interpolation_to_str = {
    cv2.INTER_NEAREST: 'INTER_NEAREST',
    cv2.INTER_LINEAR: 'INTER_LINEAR',
    cv2.INTER_AREA: 'INTER_AREA',
    cv2.INTER_CUBIC: 'INTER_CUBIC',
    cv2.INTER_LANCZOS4: 'INTER_LANCZOS4'
}


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ColorTrans(object):
    """Transpose the image color space from RGB to BGR or reverse.
    Args:
        mode (int): 0 for BGR to RGB, 1 for RGB to BGR.
    """

    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if self.mode == 0 and tensor.shape[2] == 3:
            return cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        elif self.mode == 1 and tensor.shape[2] == 3:
            return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input should be a 3 channel images.")

    def __repr__(self):
        return self.__class__.__name__ + '(mode={0})'.format(self.mode)


class Resize(object):
    """Resize the input numpy.ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, shorter=True):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.shorter = shorter

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be scaled.
        Returns:
            numpy.ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation, self.shorter)

    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomCrop(object):
    """Crop the given numpy.ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
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
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy.ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        height, width = img.shape[:2]
        th, tw = output_size
        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class CenterCrop(object):
    """Crops the given numpy.ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class FiveCrop(object):
    """Crops the five numpy.ndarray.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the numpy.ndarray belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __init__(self, norm_value=255, nchw=True):
        assert isinstance(norm_value, int)

        self.norm_value = norm_value
        self.nchw = nchw

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic, self.norm_value, self.nchw)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Pad(object):
    """Pad the given numpy.ndarray on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be padded.
        Returns:
            numpy.ndarray: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class PadTarget(object):
    """Pad the given numpy.ndarray on all sides with the given "pad" value.
    Args:
        size (int or tuple): If a single int is provided this is used to pad all borders. 
            If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively. 
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_type (str): Type of padding. Should be: lefttop, leftbottom, righttop, rightbottom or center.
            Default is center.
            - center: Padding on each border.
            - lefttop: Padding on left and top.
            - leftbottom: Padding on left and bottom.
            - righttop: Padding on right and top.
            - rightbottom: Padding on right and bottom.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, fill=0, padding_type="center", padding_mode='constant'):
        assert isinstance(size, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_type in ['lefttop', 'leftbottom', 'righttop', 'rightbottom', 'center']
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(size, Sequence) and len(size) not in [2]:
            raise ValueError("Padding must be an int or a 2 element tuple, not a " +
                             "{} element tuple".format(len(size)))

        self.size = size
        self.fill = fill
        self.padding_type = padding_type
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be padded.
        Returns:
            numpy.ndarray: Padded image.
        """

        # left, top, right and bottom
        height, width = img.shape[:2]
        if isinstance(self.size, numbers.Number):
            assert max(height, width) >= self.size
        if len(self.size) == 2:
            assert (self.size[0] - height) >= 0 and (self.size[1] - width) >= 0

        if self.padding_type == "center":
            padding_left = (self.size[1] - width)//2
            padding_top = (self.size[0] - height)//2
            padding_right = self.size[1] - padding_left - width
            padding_bottom = self.size[0] - padding_top - height
        elif self.padding_type == "lefttop":
            padding_left = self.size[1] - width
            padding_top = self.size[0] - height
            padding_right = 0
            padding_bottom = 0
        elif self.padding_type == "leftbottom":
            padding_left = self.size[1] - width
            padding_top = 0
            padding_right = 0
            padding_bottom = self.size[0] - height
        elif self.padding_type == "righttop":
            padding_left = 0
            padding_top = self.size[0] - height
            padding_right = self.size[1] - width
            padding_bottom = 0
        elif self.padding_type == "rightbottom":
            padding_left = 0
            padding_top = 0
            padding_right = self.size[1] - width
            padding_bottom = self.size[0] - height

        padding = (padding_left, padding_top, padding_right, padding_bottom)

        return F.pad(img, padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, fill={1}, padding_type={2}, padding_mode={2})'.\
            format(self.size, self.fill, self.padding_type, self.padding_mode)


class Grayscale(object):
    """Convert image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        numpy.ndarray: Grayscale version of the input.
         - If ``num_output_channels == 1`` : returned image is single channel
         - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be converted to grayscale.
        Returns:
            numpy.ndarray: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image.
        Returns:
            numpy.ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy.ndarray randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
        Returns:
            numpy.ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given numpy.ndarray randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
        Returns:
            numpy.ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
        Returns:
            numpy.ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class Rotation(object):
    """Rotate the image by angle.
    Args:
        angle (int): Rotate angle.
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
    """

    def __init__(self, angle, resample=False, expand=False, center=None, fill=None):
        if isinstance(angle, int):
            self.angle = angle
        else:
            raise ValueError("angle must be an int number.")

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
        Returns:
            numpy.ndarray: Rotated image.
        """

        return F.rotate(img, self.angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0}'.format(self.angle)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class FastRotation(object):
    """Rotate the image with 90, 180 and 270.
    Args:
        angle (int): Rotate angle, must be 90, 180 or 270.
    """

    def __init__(self, angle):
        self.candidate = [90, 180, 270]
        if angle in self.candidate:
            self.mode = self.candidate.index(angle)
        else:
            raise ValueError("angle must be an int number.")

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
        Returns:
            numpy.ndarray: Rotated image.
        """

        return F.fastrotate(img, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0}'.format(self.candidate[self.mode])
        format_string += ')'
        return format_string


class ConditionRotation(object):
    """Rotate the image to same width/height ratio.
    Args:
        short (bool): 
            If true or omitted, the width is shorter one.
            If false, the height is shorter one.
        clockwise (bool): 
            If true or omitted, clockwise rotation.
            If false, counterclockwise rotation.
        
    """

    def __init__(self, short=True, clockwise=True):
        self.short = short
        self.clockwise = 0 if clockwise else 2

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
        Returns:
            numpy.ndarray: Rotated image.
        """

        height, width = img.shape[:2]
        if self.short:
            if height > width:
                return img
            else:
                return F.fastrotate(img, self.clockwise)
        else:
            if height > width:
                return F.fastrotate(img, self.clockwise)
            else:
                return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(short={0}, clockwise={1}'.format(self.short, self.clockwise)
        format_string += ')'
        return format_string


class RandomResizedCrop(object):
    """Crop the given numpy.ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy.ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width = img.shape[:2]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped and resized.
        Returns:
            numpy.ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class JpegCompress(object):
    def __init__(self, prob=0.3, ratio=(40, 100)):
        if ratio[0] > ratio[1]:
            warnings.warn("jpeg compress ratio range should be of kind (min, max)")
        self.prob = prob
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < self.prob:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.ratio[0], self.ratio[1])]
            img_encode = cv2.imencode('.jpg', img, encode_param)[1]
            img_compress = cv2.imdecode(np.array(img_encode), cv2.IMREAD_COLOR)
            return img_compress
        else:
            return img


    def __repr__(self):
        interpolate_str = _cv2_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(prob={0}, ratio={1})'.format(self.prob, self.ratio)



