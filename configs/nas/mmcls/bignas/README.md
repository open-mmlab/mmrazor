# BigNAS

> [BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models](https://arxiv.org/abs/2003.11142)

<!-- [ALGORITHM] -->

## Abstract

Neural architecture search (NAS) has shown promising results discovering models that are both accurate and fast. For NAS, training a one-shot model has become a popular strategy to rank the relative quality of different architectures (child models) using a single set of shared weights. However, while one-shot model weights can effectively rank different network architectures, the absolute accuracies from these shared weights are typically far below those obtained from stand-alone training. To compensate, existing methods assume that the weights must be retrained, finetuned, or otherwise post-processed after the search is completed. These steps significantly increase the compute requirements and complexity of the architecture search and model deployment. In this work, we propose BigNAS, an approach that challenges the conventional wisdom that post-processing of the weights is necessary to get good prediction accuracies. Without extra retraining or post-processing steps, we are able to train a single set of shared weights on ImageNet and use these weights to obtain child models whose sizes range from 200 to 1000 MFLOPs. Our discovered model family, BigNASModels, achieve top1 accuracies ranging from 76.5% to 80.9%, surpassing state-of-the-art models in this range including EfficientNets and Once-for-All networks without extra retraining or post-processing. We present ablative study and analysis to further understand the proposed BigNASModels.

## Get Started

### Step 1: Supernet pre-training on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/mmcls/bignas/attentive_mobilenet_supernet_32xb64_in1k.py 4 \
  --work-dir $WORK_DIR
```

### Step 2: Search for subnet on the trained supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/mmcls/bignas/attentive_mobilenet_search_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP1_CKPT
```

### Step 3: Subnet inference on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/mmcls/bignas/attentive_mobilenet_subnet_8xb256_in1k.py \
  $STEP2_CKPT 1 --work-dir $WORK_DIR \
  --cfg-options algorithm.mutable_cfg=$STEP2_SUBNET_YAML
```

## Results and models

| Dataset  |       Supernet       |                                                                                                                   Subnet                                                                                                                    |      Params(M)       |       Flops(G)       |          Top-1          |                           Config                            |                                                                                                                                                                                                                                                                                                               Download                                                                                                                                                                                                                                                                                                               |         Remarks         |
| :------: | :------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: | :------------------: | :---------------------: | :---------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------: |
| ImageNet | AttentiveMobileNetV3 |      [mutable](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml)      | 8.9(min) / 23.3(max) | 203(min) / 1939(max) | 77.25(min) / 81.72(max) | [config](./detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py) | [pretrain](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.log.json) |    MMRazor searched     |
| ImageNet | AttentiveMobileNetV3 | [AttentiveNAS-A0\*](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml) |        11.559        |         414          |         77.252          | [config](./detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py) | [pretrain](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.log.json) | Converted from the repo |
| ImageNet | AttentiveMobileNetV3 | [AttentiveNAS-A6\*](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml) |        16.476        |         1163         |         80.790          | [config](./detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py) | [pretrain](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.log.json) | Converted from the repo |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/AttentiveNAS). The config files of these models
are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

**Note**: In the official `AttentiveNAS` code, the `AutoAugmentation` in Calib-BN subnet recommended to use large batchsize to evaluation like `256`, which leads to higher performance. Compared with the original configuration file, this configuration has been modified as follows:

- modified the settings related to `batchsize` in `train_pipeline` and `test_pipeline`, e.g. setting `train_dataloader.batch_size=256`、 `val_dataloader.batch_size=256`、`test_cfg.calibrate_sample_num=16384` and `collate_fn=dict(type='default_collate')` in train_dataloader.
- setting `dict(type='mmrazor.AutoAugment', policies='original')` instead of `dict(type='mmrazor.AutoAugmentV2', policies=policies)` in train_pipeline.

1. Used search_space in AttentiveNAS, which is different from BigNAS paper.
2. The Top-1 Acc is unstable and may fluctuate by about 0.1, convert the official weight according to the [converter script](../../../../tools/model_converters/convert_attentivenas_nas_ckpt.py). A Calib-BN model will be released later.
3. We have observed that the searchable model has been officially released. Although the subnet accuracy has decreased, it is more efficient. We will also provide the supernet training configuration in the future.

## Citation

```latex
@inproceedings{yu2020bignas,
  title={Bignas: Scaling up neural architecture search with big single-stage models},
  author={Yu, Jiahui and Jin, Pengchong and Liu, Hanxiao and Bender, Gabriel and Kindermans, Pieter-Jan and Tan, Mingxing and Huang, Thomas and Song, Xiaodan and Pang, Ruoming and Le, Quoc},
  booktitle={European Conference on Computer Vision},
  pages={702--717},
  year={2020},
  organization={Springer}
}
```
