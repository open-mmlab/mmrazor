# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List
from typing import Dict, Callable
from mmrazor.registry import MODELS
from mmengine.config import Config
import os
from mmengine.utils import get_installed_path
from mmrazor.registry import MODELS
import torch
import torch.nn as nn
from .models import (AddCatModel, ConcatModel, ConvAttnModel, DwConvModel,
                     ExpandLineModel, GroupWiseConvModel, SingleLineModel,
                     MultiBindModel, MultiConcatModel, MultiConcatModel2,
                     ResBlock, Xmodel, MultipleUseModel, Icep, SelfAttention)
import json
# model generator
from mmdet.testing._utils import demo_mm_inputs
import string
import copy
# helper functions


def get_shape(tensor, only_length=False):
    if isinstance(tensor, torch.Tensor):
        if only_length:
            return len(tensor.shape)
        else:
            return tensor.shape
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        shapes = []
        for x in tensor:
            shapes.append(get_shape(x, only_length))
        return shapes
    elif isinstance(tensor, dict):
        shapes = {}
        for key in tensor:
            shapes[key] = get_shape(tensor[key], only_length)
        return shapes
    else:
        raise NotImplementedError(
            f'unsuppored type{type(tensor)} to get shape of tensors.')


# generators


class ModelGenerator(nn.Module):

    def __init__(self, name: str, model_src) -> None:
        super().__init__()
        self.name = name
        self.model_src = model_src
        self._model = None

    def __call__(self, *args, **kwargs):
        return self.init_model()

    def init_model(self):
        return self.model_src()

    def forward(self, x):
        assert self._model is not None
        return self._model(x, *self.input())

    def input(self):
        return []

    def assert_model_is_changed(self, tensors_org, tensors_new):
        shape1 = get_shape(tensors_org)
        shape2 = get_shape(tensors_new)
        assert shape1 == shape2, f'{shape1}!={shape2}'

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def get_base_name(cls, name: str):
        names = name.split('.')
        return '.'.join(names[1:])

    @classmethod
    def get_short_name(cls, name: str):
        scope = name.split('.')[0]
        base_name = cls.get_base_name(name)
        names = base_name.replace('-', '.').replace('_', '.').split('.')
        name = names[0]
        name = name.rstrip(string.digits)

        return f'{scope}.{name}'

    @property
    def base_name(self):
        return self.__class__.get_base_name(self.name)

    @property
    def short_name(self):
        return self.__class__.get_short_name(self.name)

    @property
    def scope(self):
        return self.name.split('.')[0]


class MMModelGenerator(ModelGenerator):

    def __init__(self, name, cfg) -> None:
        self.cfg = cfg
        super().__init__(name, self.get_model_src)

    def get_model_src(self):
        model = MODELS.build(self.cfg)
        model = revert_sync_batchnorm(model)
        return model

    def __repr__(self) -> str:
        return self.name


class MMDetModelGenerator(MMModelGenerator):

    def forward(self, x):
        assert self._model is not None
        self._model.eval()
        return self._model(x, **self.input(), mode='tensor')

    def input(self):
        data = demo_mm_inputs(1, [[3, 224, 224]])
        data = self._model.data_preprocessor(data, False)
        data.pop('inputs')
        return data

    def assert_model_is_changed(self, tensors_org, tensors_new):
        assert get_shape(tensors_org, True) == get_shape(tensors_new, True)


# model library


class ModelLibrary:
    default_includes: List = []
    _models = None

    def __init__(self, include=default_includes, exclude=[]) -> None:
        self.include_key = include
        self.exclude_key = exclude
        self._include_models, self._uninclude_models, self.exclude_models =\
             self._classify_models(self.models)

    @property
    def models(self):
        if self.__class__._models is None:
            self.__class__._models: Dict[
                str, Callable] = self.__class__.get_models()
        return self.__class__._models

    @classmethod
    def get_models(cls):
        raise NotImplementedError()

    def include_models(self):
        return self._include_models

    def uninclude_models(self):
        return self._uninclude_models

    def is_include(self, name: str, includes: List[str], start_with=True):
        for key in includes:
            if start_with:
                if name.startswith(key):
                    return True
            else:
                if key in name:
                    return True
        return False

    def is_default_includes_cover_all_models(self):
        models = copy.copy(self._models)
        is_covered = True
        for name in models:
            if self.is_include(name, self.__class__.default_includes):
                pass
            else:
                is_covered = False
                print(name, '\tnot include')
        return is_covered

    def short_names(self):
        short_names = set()
        for name in self.models:
            short_names.add(self.models[name].short_name)
        return short_names

    def _classify_models(self, models: Dict):
        include = []
        uninclude = []
        exclude = []
        for name in models:
            if self.is_include(name, self.exclude_key, start_with=False):
                exclude.append(models[name])
            elif self.is_include(name, self.include_key, start_with=True):
                include.append(models[name])
            else:
                uninclude.append(models[name])
        return include, uninclude, exclude

    def get_short_name_of_model(self, name: str):
        names = name.replace('-', '.').replace('_', '.').split('.')
        return names[0]


class DefaultModelLibrary(ModelLibrary):
    _mm_models = None

    default_includes: List = [
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
        'SelfAttention',
        # mm models
        'resnet',
        'pspnet',
        'yolo'
    ]

    def __init__(self,
                 include=default_includes,
                 exclude=[],
                 with_mm_models=False) -> None:
        self.with_mm_models = with_mm_models
        super().__init__(include, exclude)

    @property
    def models(self):
        models = copy.copy(super().models)
        if self.with_mm_models:
            models.update(self.mm_models)
        return models

    @property
    def mm_models(self):
        if self.__class__._mm_models is None:
            self.__class__._mm_models = self.get_mm_models()
        return self.__class__._mm_models

    @classmethod
    def get_models(cls):
        models = [
            SingleLineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            MultipleUseModel,
            Icep,
            ExpandLineModel,
            MultiBindModel,
            DwConvModel,  #
            ConvAttnModel,
            SelfAttention,
        ]
        model_dict = {}
        for model in models:
            model_dict[model.__name__] = ModelGenerator(
                'default.' + model.__name__, model)
        return model_dict

    @classmethod
    def get_mm_models(cls):
        paths = [
            'mmcls::resnet/resnet34_8xb32_in1k.py',
            'mmseg::pspnet/pspnet_r18-d8_4xb4-80k_potsdam-512x512.py',
            'mmdet::yolo/yolov3_d53_8xb8-320-273e_coco.py'
        ]
        models = {}
        for path in paths:
            Model = MMModelLibrary.get_model_from_path(path)
            models[Model.base_name] = Model
        return models


class TorchModelLibrary(ModelLibrary):

    default_includes = [
        'alexnet', 'densenet', 'efficientnet', 'googlenet', 'inception',
        'mnasnet', 'mobilenet', 'regnet', 'resnet', 'resnext', 'shufflenet',
        'squeezenet', 'vgg', 'wide_resnet', "vit", "swin", "convnext"
    ]

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(include, exclude)

    @classmethod
    def get_models(cls):
        from inspect import isfunction

        import torchvision

        attrs = dir(torchvision.models)
        models = {}
        for name in attrs:
            module = getattr(torchvision.models, name)
            if isfunction(module) and name is not 'get_weight':
                models[name] = ModelGenerator('torch.' + name, module)
        return models


class MMModelLibrary(ModelLibrary):
    default_includes = []
    base_config_path = '/'
    repo = 'mmxx'

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(include, exclude)

    @classmethod
    def scope_path(cls):
        path = cls._scope_path(cls.repo) + cls.base_config_path
        return path

    @classmethod
    def get_models(cls):
        models = {}
        added_models = set()
        for dirpath, dirnames, filenames in os.walk(cls.scope_path()):
            for filename in filenames:
                if filename.endswith('.py'):

                    cfg_path = dirpath + '/' + filename
                    try:
                        config = Config.fromfile(cfg_path)
                    except:
                        continue
                    if 'model' in config:

                        # get model_name
                        model_name = cls.get_model_name_from_path(
                            cfg_path, cls.scope_path())

                        model_cfg = config['model']
                        model_cfg = cls._config_process(model_cfg)
                        if json.dumps(model_cfg) not in added_models:
                            models[model_name] = cls.generator_type()(
                                cls.repo + '.' + model_name, model_cfg)
                            added_models.add(json.dumps(model_cfg))
        return models

    @classmethod
    def generator_type(cls):
        return MMModelGenerator

    @classmethod
    def get_model_name_from_path(cls, config_path, scope_path):
        import os
        dirpath = os.path.dirname(config_path) + '/'
        filename = os.path.basename(config_path)

        model_type_name = '_'.join(dirpath.replace(scope_path, '').split('/'))
        model_type_name = model_type_name if model_type_name == '' else model_type_name + '_'
        model_name = model_type_name + \
            os.path.basename(filename).split('.')[0]
        return model_name

    @classmethod
    def get_model_from_path(cls, config_path):
        path, scope = Config._get_cfg_path(config_path, '')
        if scope is None:
            scope = 'mmrazor'
        config = Config.fromfile(path)['model']
        config = cls._config_process(config=config)
        config['_scope_'] = scope
        name = cls.get_model_name_from_path(path, cls._scope_path(scope))
        return cls.generator_type()(scope + '.' + name, config)

    @staticmethod
    def _scope_path(scope):
        if scope == 'mmseg':
            scope = 'mmsegmentation'
        repo_path = get_installed_path(scope)
        path = repo_path + '/.mim/configs/'
        return path

    @classmethod
    def _config_process(cls, config: Dict):
        config['_scope_'] = cls.repo
        config = cls._remove_certain_key(config, 'init_cfg')
        config = cls._remove_certain_key(config, 'pretrained')
        config = cls._remove_certain_key(config, 'Pretrained')
        return config

    @classmethod
    def _remove_certain_key(cls, config: Dict, key: str = 'init_cfg'):
        if isinstance(config, dict):
            if key in config:
                config.pop(key)
            for keyx in config:
                config[keyx] = cls._remove_certain_key(config[keyx], key)
        return config


class MMClsModelLibrary(MMModelLibrary):

    default_includes = [
        'vgg',
        'efficientnet',
        'resnet',
        'mobilenet',
        'resnext',
        'wide-resnet',
        'shufflenet',
        'hrnet',
        'resnest',
        'inception',
        'res2net',
        'densenet',
        'convnext',
        'regnet',
        'van',
        'swin_transformer',
        'convmixer',
        't2t',
        'twins',
        'repmlp',
        'tnt',
        't2t',
        'mlp_mixer',
        'conformer',
        'poolformer',
        'vit',
        'efficientformer',
        'mobileone',
        'edgenext',
        'mvit',
        'seresnet',
        'repvgg',
        'seresnext',
        'deit',
        'replknet',
        'hornet',
        'mobilevit',
        'davit',
    ]
    base_config_path = '_base_/models/'
    repo = 'mmcls'

    def __init__(
            self,
            include=default_includes,
            exclude=['cutmix', 'cifar', 'gem', 'efficientformer']) -> None:
        super().__init__(include=include, exclude=exclude)


class MMDetModelLibrary(MMModelLibrary):

    default_includes = [
        '_base',
        'gfl',
        'sparse',
        'simple',
        'pisa',
        'lvis',
        'carafe',
        'selfsup',
        'solo',
        'ssd',
        'res2net',
        'yolof',
        'reppoints',
        'htc',
        'groie',
        'dyhead',
        'grid',
        'soft',
        'swin',
        'regnet',
        'gcnet',
        'ddod',
        'instaboost',
        'point',
        'vfnet',
        'pafpn',
        'ghm',
        'mask',
        'resnest',
        'tood',
        'detectors',
        'cornernet',
        'convnext',
        'cascade',
        'paa',
        'detr',
        'rpn',
        'ld',
        'lad',
        'ms',
        'faster',
        'centripetalnet',
        'gn',
        'dcnv2',
        'legacy',
        'panoptic',
        'strong',
        'fpg',
        'deformable',
        'free',
        'scratch',
        'openimages',
        'fsaf',
        'rtmdet',
        'solov2',
        'yolact',
        'empirical',
        'centernet',
        'hrnet',
        'guided',
        'deepfashion',
        'fast',
        'mask2former',
        'retinanet',
        'autoassign',
        'gn+ws',
        'dcn',
        'yolo',
        'foveabox',
        'libra',
        'double',
        'queryinst',
        'resnet',
        'nas',
        'sabl',
        'fcos',
        'scnet',
        'maskformer',
        'pascal',
        'cityscapes',
        'timm',
        'seesaw',
        'pvt',
        'atss',
        'efficientnet',
        'wider',
        'tridentnet',
        'dynamic',
        'yolox',
        'albu',
        'misc',
        'crowddet',
        'condins',
    ]
    base_config_path = '/'
    repo = 'mmdet'

    def __init__(
        self,
        include=default_includes,
        exclude=[
            'lad',
            'ld',
            'faster_rcnn_faster-rcnn_r50-caffe-c4_ms-1x_coco',
        ]
    ) -> None:
        super().__init__(include=include, exclude=exclude)

    @classmethod
    def _config_process(cls, config: Dict):
        config = super()._config_process(config)
        if 'preprocess_cfg' in config:
            config.pop('preprocess_cfg')
        return config

    @classmethod
    def generator_type(cls):
        return MMModelGenerator


class MMSegModelLibrary(MMModelLibrary):
    default_includes: List = [
        '_base_',
        'knet',
        'sem',
        'dnlnet',
        'dmnet',
        'icnet',
        'apcnet',
        'swin',
        'isanet',
        'fastfcn',
        'poolformer',
        'mae',
        'segformer',
        'ccnet',
        'twins',
        'emanet',
        'upernet',
        'beit',
        'hrnet',
        'bisenetv2',
        'vit',
        'setr',
        'cgnet',
        'ocrnet',
        'ann',
        'erfnet',
        'point',
        'bisenetv1',
        'nonlocal',
        'unet',
        'danet',
        'stdc',
        'fcn',
        'encnet',
        'resnest',
        'mobilenet',
        'convnext',
        'deeplabv3',
        'pspnet',
        'gcnet',
        'fastscnn',
        'segmenter',
        'dpt',
        'deeplabv3plus',
        'psanet',
    ]
    base_config_path = '/'
    repo = 'mmsegmentation'

    def __init__(self, include=default_includes, exclude=['_base_']) -> None:
        super().__init__(include, exclude)

    @classmethod
    def _config_process(cls, config: Dict):
        config['_scope_'] = 'mmseg'
        return config


# tools


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = nn.BatchNorm2d
        module_output = nn.BatchNorm2d(module.num_features, module.eps,
                                       module.momentum, module.affine,
                                       module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
