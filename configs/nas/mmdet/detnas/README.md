# DetNAS

> [DetNAS: Backbone Search for Object Detection](https://arxiv.org/abs/1903.10979)

<!-- [ALGORITHM] -->

## Abstract

Object detectors are usually equipped with backbone networks designed for image classification. It might be sub-optimal because of the gap between the tasks of image classification and object detection. In this work, we present DetNAS to use Neural Architecture Search (NAS) for the design of better backbones for object detection. It is non-trivial because detection training typically needs ImageNet pre-training while NAS systems require accuracies on the target detection task as supervisory signals. Based on the technique of one-shot supernet, which contains all possible networks in the search space, we propose a framework for backbone search on object detection. We train the supernet under the typical detector training schedule: ImageNet pre-training and detection fine-tuning. Then, the architecture search is performed on the trained supernet, using the detection task as the guidance. This framework makes NAS on backbones very efficient. In experiments, we show the effectiveness of DetNAS on various detectors, for instance, one-stage RetinaNet and the two-stage FPN. We empirically find that networks searched on object detection shows consistent superiority compared to those searched on ImageNet classification. The resulting architecture achieves superior performance than hand-crafted networks on COCO with much less FLOPs complexity.

![pipeline](https://user-images.githubusercontent.com/88702197/187425296-64baa22a-9422-46cd-bd95-47e3e5707f75.jpg)

## Get Started

### Step 1: Supernet pre-training on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/detnas/detnas_supernet_shufflenetv2_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR
```

### Step 2: Supernet fine-tuning on COCO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/detnas/detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP1_CKPT
```

### Step 3: Search for subnet on the trained supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/detnas/detnas_evolution_search_frcnn_shufflenetv2_fpn_coco.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP2_CKPT
```

### Step 4: Subnet retraining on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/detnas/detnas_subnet_shufflenetv2_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR --cfg-options algorithm.mutable_cfg=$STEP3_SUBNET_YAML  # or modify the config directly
```

### Step 5: Subnet fine-tuning on COCO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py 4 \
  --work-dir $WORK_DIR \
  --cfg-options --cfg-options model.init_cfg.checkpoint=$STEP4_CKPT
```

### Step 6: Subnet inference on COCO

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py \
  none 1 --work-dir $WORK_DIR \
  --cfg-options model.init_cfg.checkpoint=$STEP5_CKPT
```

## Results and models

| Dataset |      Supernet      |                                                                                                              Subnet                                                                                                               |   Params(M)    |    Flops(G)    | mAP  |                           Config                            |                                                                                                                                                                                                                                                                                                               Download                                                                                                                                                                                                                                                                                                               |     Remarks      |
| :-----: | :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------: | :------------: | :--: | :---------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: |
|  COCO   | FRCNN-ShuffleNetV2 | [mutable](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml) | 3.35(backbone) | 0.34(backbone) | 37.5 | [config](./detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco.py) | [pretrain](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.log.json) | MMRazor searched |

**Note**:

1. The experiment settings of DetNAS are similar with SPOS's, and our training dataset is COCO2017 rather than COCO2014.
2. We also retrained official subnet with same experiment settings, the final result is 36.9

## Citation

```latex
@article{chen2019detnas,
  title={Detnas: Backbone search for object detection},
  author={Chen, Yukang and Yang, Tong and Zhang, Xiangyu and Meng, Gaofeng and Xiao, Xinyu and Sun, Jian},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={6642--6652},
  year={2019}
}
```
