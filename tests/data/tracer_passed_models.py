# Copyright (c) OpenMMLab. All rights reserved.
from .model_library import (MMClsModelLibrary, MMDetModelLibrary,
                            DefaultModelLibrary, TorchModelLibrary,
                            MMPoseModelLibrary, MMSegModelLibrary)


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
    _mmpose_library = None

    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
                self.__class__.mmpose_library(),
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
                # mm models
                'resnet',
                'pspnet',
                'yolo'
            ],with_mm_models=True)

        return cls._default_library

    @classmethod
    def torch_library(cls):
        """
        googlenet: return a tuple when training, so it should
        trace in eval mode
        """
        torch_includes = [
            'resnext',
            'efficientnet',
            'inception',
            'wide',
            'resnet',
            'regnet',
            'shufflenet',
            'mnasnet',
            'vit',
            'convnext',
            'googlenet',
            'densenet',
            'swin',
            'vgg',
            'mobilenet',
            'squeezenet',
            'alexnet',
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
            'tnt',
            'resnet',
            'resnetv1c',
            'mobileone',
            'mlp',
            'densenet',
            'hrnet',
            'seresnet',
            'van',
            'repmlp',
            'repvgg',
            'vgg',
            'vgg11bn',
            'edgenext',
            'vgg19bn',
            'wide',
            'res2net',
            'vgg13bn',
            'resnetv1d',
            'mobilenet',
            'convmixer',
            'resnest',
            'inception',
            'resnext',
            'twins',
            'vgg16bn',
            'shufflenet',
            'conformer',
            'regnet',
            'seresnext',
            'vit',
            'poolformer',
            't2t',
            'efficientnet',
            ## error
            # 'deit',
            # 'swin',
            # 'convnext',
            # 'mvit'
        ]
        if cls._mmcls_library is None:
            cls._mmcls_library = MMClsModelLibrary(include=mmcls_include)
        return cls._mmcls_library

    @classmethod
    def mmdet_library(cls):
        mmdet_include = [
            'pafpn',
            'gn+ws',
            'paa',
            'fcos',
            'autoassign',
            'centripetalnet',
            'retinanet',
            'cornernet',
            'gn',
            'instaboost',
            'rpn',
            'fpg',
            'crowddet',
            'resnest',
            'pvt',
            'solo',
            'grid',
            'free',
            'point',
            'yolo',
            'double',
            'dynamic',
            'maskformer',
            'scratch',
            'nas',
            'yolof',
            'faster',
            'atss',
            'yolox',
            'fsaf',
            'ghm',
            'centernet',
            'seesaw',
            'regnet',
            'cityscapes',
            'lvis',
            'sabl',
            'gfl',
            'tridentnet',
            'selfsup',
            'deepfashion',
            'efficientnet',
            'foveabox',
            'mask',
            ## errors
            # 'timm',
            # 'swin',
            # 'dyhead',
            # 'hrnet',
            # 'deformable',
            # 'ssd',
            # 'empirical',
            # 'detectors',
            # 'reppoints',
            # 'scnet',
            # 'legacy',
            # 'htc',
            # 'dcnv',
            # 'carafe',
            # 'yolact',
            # 'panoptic',
            # 'misc',
            # 'rtmdet',
            # 'pascal',
            # 'ddod',
            # 'mask2former',
            # 'tood',
            # 'queryinst',
            # 'simple',
            # 'pisa',
            # 'fast',
            # 'cascade',
            # 'wider',
            # 'openimages',
            # '',
            # 'strong',
            # 'res2net',
            # 'libra',
            # 'vfnet',
            # 'soft',
            # 'sparse',
            # 'gcnet',
            # 'convnext',
            # 'ms',
            # 'dcn',
            # 'guided',
            # 'groie',
            # 'solov',
            # 'detr',
        ]
        if cls._mmdet_library is None:
            cls._mmdet_library = MMDetModelLibrary(mmdet_include)
        return cls._mmdet_library

    @classmethod
    def mmseg_library(cls):
        # a common error: unet related models
        include = [
            'bisenetv',
            'erfnet',
            'dmnet',
            'twins',
            'segformer',
            'isanet',
            'vit',
            'resnest',
            'setr',
            'cgnet',
            'stdc',
            'dpt',
            'pspnet',
            'upernet',
            'apcnet',
            'gcnet',
            'ann',
            'ocrnet',
            'ccnet',
            'deeplabv',
            'dnlnet',
            'point',
            'fastscnn',
            'psanet',
            'segmenter',
            'danet',
            'emanet',
            'icnet',
            'unet',
            'fcn',
            'swin',
            'nonlocal',
            'deeplabv3plus',
            'sem',
            ## errors
            # 'mobilenet',
            # 'mae',
            # 'knet',
            # 'poolformer',
            # 'beit',
            # 'encnet',
            # 'hrnet',
            # 'convnext',
            # 'fastfcn'
        ]
        if cls._mmseg_library is None:
            cls._mmseg_library = MMSegModelLibrary(include=include)
        return cls._mmseg_library


    @classmethod
    def mmpose_library(cls):
        mmpose_include = [
            'hand',
            'face',
            'wholebody',
            'body',
            'animal',
        ]
        if cls._mmpose_library is None:
            cls._mmpose_library = MMPoseModelLibrary(include=mmpose_include)
        
        return cls._mmpose_library

    # for backward tracer


class BackwardPassedModelManager(PassedModelManager):

    _default_library = None
    _torch_library = None
    _mmcls_library = None
    _mmseg_library = None
    _mmdet_library = None
    _mmpose_library = None


    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
                self.__class__.mmpose_library(),
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

    @classmethod
    def mmpose_library(cls):
        mmpose_include = [
            'hand',
            'face',
            'wholebody',
            'body',
            'animal',
        ]

        if cls._mmpose_library is None:
            cls._mmpose_library = MMPoseModelLibrary(include=mmpose_include)
        return cls._mmpose_library


fx_passed_library = FxPassedModelManager()
backward_passed_library = BackwardPassedModelManager()
