# Copyright (c) OpenMMLab. All rights reserved.
from .model_library import (MMClsModelLibrary, MMDetModelLibrary, ModelLibrary,
                            DefaultModelLibrary, TorchModelLibrary,
                            MMSegModelLibrary)


class PassedModelManager:

    def __init__(self) -> None:
        pass

    def include_models(self, full_test=False):
        models = []
        for library in self.libraries(full_test):
            models.extend(library.include_models())
        return models

    def uninclude_models(self, full_test=False):
        models = []
        for library in self.libraries(full_test):
            models.extend(library.uninclude_models())
        return models

    def libraries(self, full=False):
        return []


class FxPassedModelManager(PassedModelManager):

    _default_library = None
    _torch_library = None
    _mmcls_library = None
    _mmseg_library = None
    _mmdet_library = None

    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
            ]
        else:
            return [self.__class__.default_library()]

    @classmethod
    def default_library(cls):
        if cls._default_library is None:
            cls._default_library = DefaultModelLibrary(include=[
                'SingleLineModel',
                'ResBlock',
                'AddCatModel',
                'ConcatModel',
                'MultiConcatModel',
                'MultiConcatModel2',
                'GroupWiseConvModel',
                'Xmodel',
                'MultipleUseModel',
                'Icep',
                'ExpandLineModel',
                'MultiBindModel',
                'DwConvModel',
                'ConvAttnModel',
            ])

        return cls._default_library

    @classmethod
    def torch_library(cls):
        """
        googlenet: return a tuple when training, so it should
        trace in eval mode
        """
        torch_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            'squeezenet',
            'vgg',
            'wide_resnet',
            "vit",
            "swin",
            "convnext",
            # error
            # 'shufflenet',  # bug
        ]
        if cls._torch_library is None:
            cls._torch_library = TorchModelLibrary(include=torch_includes)
        return cls._torch_library

    @classmethod
    def mmcls_library(cls):
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            'hrnet',
            'inception',
            'densenet',
            'regnet',
            'convmixer',
            'efficientformer',
            'mobileone',
            'edgenext',
            'seresnet',
            'repvgg',
            'seresnext',
            'conformer',
            'poolformer',
            'res2net',
            'resnest',
            'convnext',
            # errors
            # 'mvit', # error
            # 'van',  # bug
            # 'twins',  # bug
            # 'tnt',  # bug
            # 'repmlp',  # bug
            # 't2t',  # bug
            # 'swin',  # bug
            # 'shufflenet',  # bug
            # 'vit',  # bug
            # 'mlp',  # bug
        ]
        if cls._mmcls_library is None:
            cls._mmcls_library = MMClsModelLibrary(include=mmcls_include)
        return cls._mmcls_library

    @classmethod
    def mmdet_library(cls):
        mmdet_include = [
            'retinanet',
            'faster_rcnn',
            'mask_rcnn',
            'fcos',
            'yolo',
            'gfl',
            'simple',
            'lvis',
            'selfsup',
            'solo',
            'soft',
            'instaboost',
            'point',
            'pafpn',
            'ghm',
            'paa',
            'rpn',
            'faster',
            'centripetalnet',
            'gn',
            'free',
            'scratch',
            'centernet',
            'deepfashion',
            'autoassign',
            'gn+ws',
            'foveabox',
            'resnet',
            'cityscapes',
            'timm',
            'atss',
            'dynamic',
            'panoptic',
            'solov2',
            'fsaf',
            'double',
            'cornernet',
            # 'vfnet',  # error
            # 'carafe',  # error
            # 'sparse',  # error
            # '_base', # error
            # 'ssd', # error
            # 'res2net',  # error
            # 'reppoints', # error
            # 'groie',  # error
            # 'dyhead',  # error
            # 'ms', # error
            # 'detr',  # error
            # 'swin', # error
            # 'regnet',  # error
            # 'gcnet', # error
            # 'ddod', # error
            # 'resnest',  # error
            # 'tood',  # error
            # 'cascade', # error
            # 'dcnv2',  # error
            # 'strong',  # error
            # 'fpg',  # error
            # 'deformable',  # error
            # 'mask2former',  # error
            # 'hrnet',  # error
            # 'guided',  # error
            # 'nas',  # error
            # 'yolact',  # error
            # 'empirical',  # error
            # 'dcn',  # error
            # 'fast', # error
            # 'queryinst',  # error
            # 'pascal',  # error
            # 'efficientnet',  # error
            # 'tridentnet',  # error
            # 'rtmdet', # error
            # 'seesaw', # error
            # 'pvt',# error
            # 'detectors',# error
            # 'htc',# error
            # 'wider',# error
            # 'maskformer',# error
            # 'grid',# error
            # 'openimages',# error
            # 'legacy',# error
            # 'pisa',# error
            # 'libra',# error
            # 'convnext',# error
            # 'scnet',# error
            # 'sabl',# error
        ]
        if cls._mmdet_library is None:
            cls._mmdet_library = MMDetModelLibrary(mmdet_include)
        return cls._mmdet_library

    @classmethod
    def mmseg_library(cls):
        # a common error: unet related models
        include = [
            'deeplabv3plus',
            # '_base_',
            # 'knet',
            # 'sem',
            # 'dnlnet',
            # 'dmnet',
            # 'icnet',
            # 'apcnet',
            # 'swin',
            # 'isanet',
            # 'fastfcn',
            # 'poolformer',
            # 'mae',
            # 'segformer',
            # 'ccnet',
            # 'twins',
            # 'emanet',
            # 'upernet',
            # 'beit',
            # 'hrnet',
            # 'bisenetv2',
            # 'vit',
            # 'setr',
            # 'cgnet',
            # 'ocrnet',
            # 'ann',
            # 'erfnet',
            # 'point',
            # 'bisenetv1',
            # 'nonlocal',
            # 'unet',
            # 'danet',
            # 'stdc',
            # 'fcn',
            # 'encnet',
            # 'resnest',
            # 'mobilenet',
            # 'convnext',
            # 'deeplabv3',
            # 'pspnet',
            # 'gcnet',
            # 'fastscnn',
            # 'segmenter',
            # 'dpt',
            # 'psanet',
        ]
        if cls._mmseg_library is None:
            cls._mmseg_library = MMSegModelLibrary(include=include)
        return cls._mmseg_library

    # for backward tracer


class BackwardPassedModelManager(PassedModelManager):

    _default_library = None
    _torch_library = None
    _mmcls_library = None
    _mmseg_library = None
    _mmdet_library = None

    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
            ]
        else:
            return [self.__class__.default_library()]

    @classmethod
    def default_library(cls):
        if cls._default_library is None:
            cls._default_library = DefaultModelLibrary(include=[
                'SingleLineModel',
                'ResBlock',
                'AddCatModel',
                'ConcatModel',
                'MultiConcatModel',
                'MultiConcatModel2',
                'GroupWiseConvModel',
                'Xmodel',
                # 'MultipleUseModel', # bug
                'Icep',
                'ExpandLineModel',
                'MultiBindModel',
                'DwConvModel',
                'ConvAttnModel',
            ])
        return cls._default_library

    @classmethod
    def torch_library(cls):
        """
        googlenet return a tuple when training, so it
            should trace in eval mode
        """

        torch_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet',     # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
            # "vit",
            # "swin",
            # "convnext"
        ]
        if cls._torch_library is None:
            cls._torch_library = TorchModelLibrary(include=torch_includes)
        return cls._torch_library

    @classmethod
    def mmcls_library(cls):
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_model_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            # 'shufflenet',  # bug
            'hrnet',
            # 'resnest',  # bug
            'inception',
            # 'res2net',  # bug
            'densenet',
            # 'convnext',  # bug
            'regnet',
            # 'van',  # bug
            # 'swin_transformer',  # bug
            # 'convmixer', # bug
            # 't2t',  # bug
            # 'twins',  # bug
            # 'repmlp',  # bug
            # 'tnt',  # bug
            # 't2t',  # bug
            # 'mlp_mixer',  # bug
            # 'conformer',  # bug
            # 'poolformer',  # bug
            # 'vit',  # bug
            # 'efficientformer',
            # 'mobileone',
            # 'edgenext'
        ]
        mmcls_exclude = ['cutmix', 'cifar', 'gem']
        if cls._mmcls_library is None:
            cls._mmcls_library = MMClsModelLibrary(
                include=mmcls_model_include, exclude=mmcls_exclude)
        return cls._mmcls_library

    @classmethod
    def mmdet_library(cls):
        mmdet_include = [
            # 'rpn',  #
            # 'faster-rcnn',
            # 'cascade-rcnn',
            # 'fast-rcnn',  # mmdet has bug.
            # 'retinanet',
            # 'mask-rcnn',
            # 'ssd300'
        ]
        if cls._mmdet_library is None:
            cls._mmdet_library = MMDetModelLibrary(mmdet_include)
        return cls._mmdet_library

    @classmethod
    def mmseg_library(cls):
        include = [
            # 'cgnet',
            # 'gcnet',
            # 'setr',
            # 'deeplabv3',
            # 'twins',
            # 'fastfcn',
            # 'fpn',
            # 'upernet',
            # 'dnl',
            # 'icnet',
            # 'segmenter',
            # 'encnet',
            # 'erfnet',
            # 'segformer',
            # 'apcnet',
            # 'fast',
            # 'ocrnet',
            # 'lraspp',
            # 'dpt',
            # 'fcn',
            # 'psanet',
            # 'bisenetv2',
            # 'pointrend',
            # 'ccnet',
            'pspnet',
            # 'dmnet',
            # 'stdc',
            # 'ann',
            # 'nonlocal',
            # 'isanet',
            # 'danet',
            # 'emanet',
            # 'deeplabv3plus',
            # 'bisenetv1',
        ]
        if cls._mmseg_library is None:
            cls._mmseg_library = MMSegModelLibrary(include=include)
        return cls._mmseg_library


fx_passed_library = FxPassedModelManager()
backward_passed_library = BackwardPassedModelManager()
