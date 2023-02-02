# Training Compact CNNs for Image Classification using Dynamic-coded Filter Fusion

## Abstract

The mainstream approach for filter pruning is usually either to force a hard-coded importance estimation upon a computation-heavy pretrained model to select “important” filters, or to impose a hyperparameter-sensitive sparse constraint on the loss objective to regularize the network training. In this paper, we present a novel filter pruning method, dubbed dynamic-coded filter fusion (DCFF), to derive compact CNNs in a computationeconomical and regularization-free manner for efficient image classification. Each filter in our DCFF is firstly given an intersimilarity distribution with a temperature parameter as a filter proxy, on top of which, a fresh Kullback-Leibler divergence based dynamic-coded criterion is proposed to evaluate the filter importance. In contrast to simply keeping high-score filters in other methods, we propose the concept of filter fusion, i.e., the weighted averages using the assigned proxies, as our preserved filters. We obtain a one-hot inter-similarity distribution as the temperature parameter approaches infinity. Thus, the relative importance of each filter can vary along with the training of the compact CNN, leading to dynamically changeable fused filters without both the dependency on the pretrained model and the introduction of sparse constraints. Extensive experiments on classification benchmarks demonstrate the superiority of our DCFF over the compared counterparts. For example, our DCFF derives a compact VGGNet-16 with only 72.77M FLOPs and 1.06M parameters while reaching top-1 accuracy of 93.47% on CIFAR-10. A compact ResNet-50 is obtained with 63.8% FLOPs and 58.6% parameter reductions, retaining 75.60% top1 accuracy on ILSVRC-2012.

![pipeline](https://user-images.githubusercontent.com/31244134/189286581-722853ba-c6d7-4a39-b902-37995b444c71.jpg)

## Results and models

### 1. Classification

| Dataset  |   Backbone   | Params(M) | FLOPs(M) | lr_type | Top-1 (%) | Top-5 (%) |                     CPrate                      |                        Config                        |           Download           |
| :------: | :----------: | :-------: | :------: | :-----: | :-------: | :-------: | :---------------------------------------------: | :--------------------------------------------------: | :--------------------------: |
| ImageNet | DCFFResNet50 |   15.16   |   2260   |  step   |   73.96   |   91.66   | \[0.0\]+\[0.35,0.4,0.1\]\*10+\[0.3,0.3,0.1\]\*6 | [config](../../mmcls/dcff/dcff_resnet_8xb32_in1k.py) | [model](<>) \| \[log\] (\<>) |

### 2. Detection

| Dataset |   Method    |   Backbone   |  Style  | Lr schd | Params(M) | FLOPs(M) | bbox AP |                     CPrate                      |                              Config                               |           Download           |
| :-----: | :---------: | :----------: | :-----: | :-----: | :-------: | :------: | :-----: | :---------------------------------------------: | :---------------------------------------------------------------: | :--------------------------: |
|  COCO   | Faster_RCNN | DCFFResNet50 | pytorch |  step   |   33.31   |  168320  |  35.8   | \[0.0\]+\[0.35,0.4,0.1\]\*10+\[0.3,0.3,0.1\]\*6 | [config](../../mmdet/dcff/dcff_faster_rcnn_resnet50_8xb4_coco.py) | [model](<>) \| \[log\] (\<>) |

### 3. Segmentation

|  Dataset   |  Method   |    Backbone     | crop size | Lr schd | Params(M) | FLOPs(M) | mIoU  |                               CPrate                                |                                Config                                 |           Download           |
| :--------: | :-------: | :-------------: | :-------: | :-----: | :-------: | :------: | :---: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: | :--------------------------: |
| Cityscapes | PointRend | DCFFResNetV1c50 | 512x1024  |  160k   |   18.43   |  74410   | 76.75 | \[0.0, 0.0, 0.0\] + \[0.35, 0.4, 0.1\] * 10 + \[0.3, 0.3, 0.1\] * 6 | [config](../../mmseg/dcff/dcff_pointrend_resnet50_8xb2_cityscapes.py) | [model](<>) \| \[log\] (\<>) |

### 4. Pose

| Dataset |     Method      |   Backbone   | crop size | total epochs | Params(M) | FLOPs(M) |  AP  |                           CPrate                           |                              Config                               |           Download           |
| :-----: | :-------------: | :----------: | :-------: | :----------: | :-------: | :------: | :--: | :--------------------------------------------------------: | :---------------------------------------------------------------: | :--------------------------: |
|  COCO   | TopDown HeatMap | DCFFResNet50 |  256x192  |     300      |   26.95   |   4290   | 68.3 | \[0.0\] + \[0.2, 0.2, 0.1\] * 10 + \[0.15, 0.15, 0.1\] * 6 | [config](../../mmpose/dcff/dcff_topdown_heatmap_resnet50_coco.py) | [model](<>) \| \[log\] (\<>) |

## Citation

```latex
@article{lin2021training,
  title={Training Compact CNNs for Image Classification using Dynamic-coded Filter Fusion},
  author={Lin, Mingbao and Ji, Rongrong and Chen, Bohong and Chao, Fei and Liu, Jianzhuang and Zeng, Wei and Tian, Yonghong and Tian, Qi},
  journal={arXiv preprint arXiv:2107.06916},
  year={2021}
}
```

## Get Started

### Generate channel_config file

Generate `resnet_pose.json` with `tools/pruning/get_channel_units.py`.

```bash
python tools/pruning/get_channel_units.py
  configs/pruning/mmpose/dcff/dcff_topdown_heatmap_resnet50.py \
  -c -i --output-path=configs/pruning/mmpose/dcff/resnet_pose.json
```

Then set layers' pruning rates `target_pruning_ratio` by `resnet_pose.json`.

### Train DCFF

#### Pose

##### COCO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/pruning/mmpose/dcff/dcff_topdown_heatmap_resnet50.py 4 \
  --work-dir $WORK_DIR
```

### Test DCFF

#### Pose

##### COCO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_test.sh \
  configs/pruning/mmpose/dcff/dcff_compact_topdown_heatmap_resnet50.py \
  $CKPT 1 --work-dir $WORK_DIR
```
