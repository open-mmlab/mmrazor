# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
from mmcls.apis import inference_model
from mmdet.apis import inference_detector
from mmseg.apis import inference_segmentor

from mmrazor.apis import init_mmcls_model, init_mmdet_model, init_mmseg_model


def _sync_bn2bn(config: mmcv.Config) -> None:

    def dfs(cfg_dict) -> None:
        if isinstance(cfg_dict, dict):
            for k, v in cfg_dict.items():
                if k == 'norm_cfg':
                    if v['type'] == 'SyncBN':
                        v['type'] = 'BN'
                dfs(v)

    dfs(config._cfg_dict)


def test_init_mmcls_model() -> None:
    from mmcls.datasets import ImageNet

    config_file = 'configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py'
    config = mmcv.Config.fromfile(config_file)
    config.model = None
    # Replace SyncBN with BN to inference on CPU
    _sync_bn2bn(config)

    mutable_file = 'configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml'
    model = init_mmcls_model(
        config,
        device='cpu',
        cfg_options={'algorithm.mutable_cfg': mutable_file})
    model.CLASSES = ImageNet.CLASSES
    assert not hasattr(model, 'architecture')
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'neck')
    assert hasattr(model, 'head')

    img = mmcv.imread(Path(__file__).parent.parent / 'data/color.jpg', 'color')
    result = inference_model(model, img)
    assert isinstance(result, dict)
    assert result.get('pred_label') is not None
    assert result.get('pred_score') is not None
    assert result.get('pred_class') is not None


def test_init_mmdet_model() -> None:
    config_file = \
        'configs/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py'
    config = mmcv.Config.fromfile(config_file)
    config.model = None
    # Replace SyncBN with BN to inference on CPU
    _sync_bn2bn(config)

    mutable_file = \
        'configs/nas/detnas/DETNAS_FRCNN_SHUFFLENETV2_340M_COCO_MMRAZOR.yaml'
    model = init_mmdet_model(
        config,
        device='cpu',
        cfg_options={'algorithm.mutable_cfg': mutable_file})
    assert not hasattr(model, 'architecture')

    img = mmcv.imread(Path(__file__).parent.parent / 'data/color.jpg', 'color')
    result = inference_detector(model, img)
    assert isinstance(result, list)


def test_init_mmseg_model() -> None:
    config_file = 'configs/distill/cwd/' \
        'cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k.py'
    config = mmcv.Config.fromfile(config_file)
    config.model = None
    # Replace SyncBN with BN to inference on CPU
    _sync_bn2bn(config)

    # Enable test time augmentation
    config.data.test.pipeline[1].flip = True

    model = init_mmseg_model(config, device='cpu')
    assert not hasattr(model, 'architecture')
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'decode_head')
    assert hasattr(model, 'auxiliary_head')

    img = mmcv.imread(Path(__file__).parent.parent / 'data/color.jpg', 'color')
    result = inference_segmentor(model, img)
    assert result[0].shape == (300, 400)
