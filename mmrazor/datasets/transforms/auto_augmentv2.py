# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random

import numpy as np
import PIL
from mmcv.transforms import Compose
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

_interpolation_name_to_pil = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'nearest': Image.NEAREST,
}


@TRANSFORMS.register_module()
class AutoAugmentOp(object):
    """Base class for ops of autoaugment."""

    def __init__(self, prob, magnitude, extra_params: dict):
        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_std = 0.5

        self.kwargs = {
            'fillcolor':
            extra_params['img_mean']
            if 'img_mean' in extra_params else _FILL,  # noqa: E501
            'resample':
            extra_params['interpolation'] if 'interpolation' in extra_params
            else _interpolation_name_to_pil.values()  # noqa: E501,E131
        }
        self._get_magnitude()

    def __call__(self, results):
        return results

    def _interpolation(self, kwargs):
        interpolation = kwargs.pop('resample', Image.NEAREST)
        if isinstance(interpolation, (list, tuple)):
            return random.choice(interpolation)
        else:
            return interpolation

    def _check_args_tf(self, kwargs):
        if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
            kwargs.pop('fillcolor')
        kwargs['resample'] = self._interpolation(kwargs)

    def _get_magnitude(self):
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))
        self.magnitude = magnitude

    def _randomly_negate(self, v):
        """With 50% prob, negate the value."""
        return -v if random.random() > 0.5 else v

    def _rotate_level_to_arg(self, level):
        # range [-30, 30]
        level = (level / _MAX_LEVEL) * 30.
        level = self._randomly_negate(level)
        return (level, )

    def _enhance_level_to_arg(self, level):
        # range [0.1, 1.9]
        return ((level / _MAX_LEVEL) * 1.8 + 0.1, )

    def _shear_level_to_arg(self, level):
        # range [-0.3, 0.3]
        level = (level / _MAX_LEVEL) * 0.3
        level = self._randomly_negate(level)
        return (level, )

    def _translate_abs_level_to_arg(self, level):
        level = (level / _MAX_LEVEL) * \
                float(_HPARAMS_DEFAULT['translate_const'])
        level = self._randomly_negate(level)
        return (level, )

    def _translate_rel_level_to_arg(self, level):
        # range [-0.45, 0.45]
        level = (level / _MAX_LEVEL) * 0.45
        level = self._randomly_negate(level)
        return (level, )


@TRANSFORMS.register_module()
class ShearX(AutoAugmentOp):
    """ShearX images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._shear_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = img.transform(img.size, Image.AFFINE,
                                (1, factor, 0, 0, 1, 0), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class ShearY(AutoAugmentOp):
    """ShearY images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._shear_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, 0, factor, 1, 0), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class TranslateXRel(AutoAugmentOp):
    """TranslateXRel images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._translate_rel_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            pixels = self.level_args[0] * img.size[0]
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, pixels, 0, 1, 0), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class TranslateYRel(AutoAugmentOp):
    """TranslateYRel images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._translate_rel_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            pixels = self.level_args[0] * img.size[1]
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, 0, 0, 1, pixels), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class TranslateX(AutoAugmentOp):
    """TranslateX images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._translate_abs_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            pixels = self.level_args[0]
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, pixels, 0, 1, 0), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class TranslateY(AutoAugmentOp):
    """TranslateY images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._translate_abs_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            pixels = self.level_args[0]
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, 0, 0, 1, pixels), **kwargs)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class RotateV2(AutoAugmentOp):
    """Rotate images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._rotate_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def transform(self, x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    def __call__(self, results, **kwargs):
        if self.prob < random.random():
            return results
        degrees = self.level_args[0]
        self._check_args_tf(kwargs)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            if _PIL_VER >= (5, 2):
                img = img.rotate(degrees, **kwargs)
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
                matrix[2], matrix[5] = self.transform(
                    -rotn_center[0] - post_trans[0],
                    -rotn_center[1] - post_trans[1], matrix)
                matrix[2] += rotn_center[0]
                matrix[5] += rotn_center[1]
                img = img.transform(img.size, Image.AFFINE, matrix, **kwargs)
            else:
                img = img.rotate(degrees, resample=kwargs['resample'])
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class AutoContrastV2(AutoAugmentOp):
    """AutoContrast images."""

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageOps.autocontrast(img)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class InvertV2(AutoAugmentOp):
    """Invert images."""

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageOps.invert(img)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class EqualizeV2(AutoAugmentOp):
    """Equalize images."""

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageOps.equalize(img)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class SolarizeV2(AutoAugmentOp):
    """Solarize images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        thresh = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageOps.solarize(img, thresh)
            img = np.array(img)
            results[key] = img
        return results

    def level_fn(self, input):
        return (int((input / _MAX_LEVEL) * 256), )


@TRANSFORMS.register_module()
class SolarizeAddV2(AutoAugmentOp):
    """SolarizeAdd images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        thresh = 128
        add = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            lut = []
            for i in range(256):
                if i < thresh:
                    lut.append(min(255, i + add))
                else:
                    lut.append(i)
            if img.mode in ('L', 'RGB'):
                if img.mode == 'RGB' and len(lut) == 256:
                    lut = lut + lut + lut
                img = img.point(lut)
            img = np.array(img)
            results[key] = img
        return results

    def level_fn(self, input):
        return (int((input / _MAX_LEVEL) * 110), )


@TRANSFORMS.register_module()
class PosterizeV2(AutoAugmentOp):
    """Posterize images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        bits_to_keep = self.level_args[0]
        if bits_to_keep >= 8:
            return results
        bits_to_keep = max(1, bits_to_keep)  # prevent all 0 images
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageOps.posterize(img, bits_to_keep)
            img = np.array(img)
            results[key] = img
        return results

    def level_fn(self, input):
        return (int((input / _MAX_LEVEL) * 4) + 4, )


@TRANSFORMS.register_module()
class ContrastV2(AutoAugmentOp):
    """Contrast images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._enhance_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageEnhance.Contrast(img).enhance(factor)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class Color(AutoAugmentOp):
    """Color images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._enhance_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageEnhance.Color(img).enhance(factor)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class BrightnessV2(AutoAugmentOp):
    """Brightness images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._enhance_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageEnhance.Brightness(img).enhance(factor)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class SharpnessV2(AutoAugmentOp):
    """Sharpness images."""

    def __init__(self, prob, magnitude, extra_params: dict):
        super().__init__(prob, magnitude, extra_params)
        self.level_fn = self._enhance_level_to_arg
        self.level_args = self.level_fn(self.magnitude)

    def __call__(self, results, **__):
        if self.prob < random.random():
            return results
        factor = self.level_args[0]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = Image.fromarray(img)
            img = ImageEnhance.Sharpness(img).enhance(factor)
            img = np.array(img)
            results[key] = img
        return results


@TRANSFORMS.register_module()
class AutoAugmentV2(object):
    """Auto Augment Implementation adapted from timm:

    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, policies):
        self.policies = copy.deepcopy(policies)
        self.sub_policy = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        sub_policy = random.choice(self.sub_policy)
        results = sub_policy(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies})'
        return repr_str
