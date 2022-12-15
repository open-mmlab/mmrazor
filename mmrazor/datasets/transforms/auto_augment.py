# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random
from typing import Optional

import numpy as np
import PIL
from mmcv.transforms import BaseTransform
from PIL import Image, ImageEnhance, ImageOps

from mmrazor.registry import TRANSFORMS

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.NEAREST, Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.NEAREST)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    """ShearX images."""

    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0),
                         **kwargs)


def shear_y(img, factor, **kwargs):
    """ShearY images."""

    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0),
                         **kwargs)


def translate_x_rel(img, pct, **kwargs):
    """TranslateXRel images."""

    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                         **kwargs)


def translate_y_rel(img, pct, **kwargs):
    """TranslateYRel images."""

    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                         **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    """TranslateX images."""

    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                         **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    """TranslateY images."""

    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                         **kwargs)


def rotate(img, degrees, **kwargs):
    """Rotate images."""

    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(-rotn_center[0] - post_trans[0],
                                         -rotn_center[1] - post_trans[1],
                                         matrix)
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    """AutoContrast images."""

    return ImageOps.autocontrast(img)


def invert(img, **__):
    """Invert images."""

    return ImageOps.invert(img)


def equalize(img, **__):
    """Equalize images."""

    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    """Solarize images."""

    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """SolarizeAdd images."""

    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ('L', 'RGB'):
        if img.mode == 'RGB' and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    """Posterize images."""

    if bits_to_keep >= 8:
        return img
    bits_to_keep = max(1, bits_to_keep)  # prevent all 0 images
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    """Contrast images."""

    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    """Color images."""

    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    """Brightness images."""

    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    """Sharpness images."""

    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value."""
    return -v if random.random() > 0.5 else v


class AutoAugmentOp(BaseTransform):

    def __init__(self, name, prob, magnitude, hparams={}):
        NAME_TO_OP = {
            'AutoContrast': auto_contrast,
            'Equalize': equalize,
            'Invert': invert,
            'Rotate': rotate,
            'Posterize': posterize,
            'Posterize2': posterize,
            'Solarize': solarize,
            'SolarizeAdd': solarize_add,
            'Color': color,
            'Contrast': contrast,
            'Brightness': brightness,
            'Sharpness': sharpness,
            'ShearX': shear_x,
            'ShearY': shear_y,
            'TranslateX': translate_x_abs,
            'TranslateY': translate_y_abs,
            'TranslateXRel': translate_x_rel,
            'TranslateYRel': translate_y_rel,
        }
        self.aug_fn = NAME_TO_OP[name]
        self.prob = prob
        self.magnitude = magnitude
        # If std deviation of magnitude is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from normal dist
        # with mean magnitude and std-dev of magnitude_std.
        # NOTE This is being tested as it's not in paper or reference impl.
        self.magnitude_std = 0.5  # FIXME add arg/hparam
        self.kwargs = {
            'fillcolor':
            hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            'resample':
            hparams['interpolation']
            if 'interpolation' in hparams else _RANDOM_INTERPOLATION
        }

        self._get_magnitude(name)

    def _get_magnitude(self, name):
        if name == 'AutoContrast' or name == 'Equalize' or name == 'Invert':
            self.level_fn = self.pass_fn
        elif name == 'Rotate':
            self.level_fn = self._rotate_level_to_arg
        elif name == 'Posterize':
            self.level_fn = self._conversion0
        elif name == 'Posterize2':
            self.level_fn = self._conversion1
        elif name == 'Solarize':
            self.level_fn = self._conversion2
        elif name == 'SolarizeAdd':
            self.level_fn = self._conversion3
        elif name in ['Color', 'Contrast', 'Brightness', 'Sharpness']:
            self.level_fn = self._enhance_level_to_arg
        elif name == 'ShearX' or name == 'ShearY':
            self.level_fn = self._shear_level_to_arg
        elif name == 'TranslateX' or name == 'TranslateY':
            self.level_fn = self._translate_abs_level_to_arg2
        elif name == 'TranslateXRel' or name == 'TranslateYRel':
            self.level_fn = self._translate_rel_level_to_arg
        else:
            print('{} not recognized'.format({}))

        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))
        self.level_args = self.level_fn(magnitude)

    def _rotate_level_to_arg(self, level):
        # range [-30, 30]
        level = (level / _MAX_LEVEL) * 30.
        level = _randomly_negate(level)
        return (level, )

    def _enhance_level_to_arg(self, level):
        # range [0.1, 1.9]
        return ((level / _MAX_LEVEL) * 1.8 + 0.1, )

    def _shear_level_to_arg(self, level):
        # range [-0.3, 0.3]
        level = (level / _MAX_LEVEL) * 0.3
        level = _randomly_negate(level)
        return (level, )

    def _translate_abs_level_to_arg2(self, level):
        level = (level / _MAX_LEVEL) * float(
            _HPARAMS_DEFAULT['translate_const'])
        level = _randomly_negate(level)
        return (level, )

    def _translate_rel_level_to_arg(self, level):
        # range [-0.45, 0.45]
        level = (level / _MAX_LEVEL) * 0.45
        level = _randomly_negate(level)
        return (level, )

    def pass_fn(self, input):
        return ()

    def _conversion0(self, input):
        return (int((input / _MAX_LEVEL) * 4) + 4, )

    def _conversion1(self, input):
        return (4 - int((input / _MAX_LEVEL) * 4), )

    def _conversion2(self, input):
        return (int((input / _MAX_LEVEL) * 256), )

    def _conversion3(self, input):
        return (int((input / _MAX_LEVEL) * 110), )

    def transform(self, results):
        if self.prob < random.random():
            return results

        for key in results.get('img_fields', ['img']):
            img = Image.fromarray(results[key])
            img = self.aug_fn(img, *self.level_args, **self.kwargs)
            results[key] = np.array(img)
        return results


@TRANSFORMS.register_module()
class AutoAugment(BaseTransform):
    """Auto Augment Implementation adapted from timm: ImageNet
    auto_augment_policy is 'original': From TPU EfficientNet impl
    https://github.com/rwightman/pytorch-image-models.

    ImageNet auto_augment_policy is 'v0':
    A PyTorch implementation of : `AutoAugment: Learning Augmentation
    Policies from Data <https://arxiv.org/abs/1805.09501>`_
    """
    auto_augment_policy = {
        'original': [
            [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
            [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
            [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
            [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
            [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
            [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
            [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
            [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
            [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
            [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
            [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
            [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
            [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
            [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
            [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
            [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
        ],
        'v0': [
            [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
            [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
            [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
            [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
            [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
            [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
            [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
            [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
            [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
            [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
            [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
            [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
            [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        ]
    }

    def __init__(self, policies: str = 'original', extra_params: dict = None):
        self.policies = copy.deepcopy(self.auto_augment_policy[policies])
        extra_params = extra_params if extra_params else dict(
            translate_const=250, img_mean=_FILL)
        self.sub_policy = [[AutoAugmentOp(*a, extra_params) for a in sp]
                           for sp in self.policies]

    def transform(self, results: dict) -> Optional[dict]:
        sub_policy = random.choice(self.sub_policy)
        for op in sub_policy:
            results = op(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies})'
        return repr_str
