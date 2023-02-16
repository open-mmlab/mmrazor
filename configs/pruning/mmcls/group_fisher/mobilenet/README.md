# Group_fisher pruning

> [Group Fisher Pruning for Practical Network Compression.](https://arxiv.org/pdf/2108.00708.pdf)

## Abstract

Network compression has been widely studied since it is able to reduce the memory and computation cost during inference. However, previous methods seldom deal with complicated structures like residual connections, group/depthwise convolution and feature pyramid network, where channels of multiple layers are coupled and need to be pruned simultaneously. In this paper, we present a general channel pruning approach that can be applied to various complicated structures. Particularly, we propose a layer grouping algorithm to find coupled channels automatically. Then we derive a unified metric based on Fisher information to evaluate the importance of a single channel and coupled channels. Moreover, we find that inference speedup on GPUs is more correlated with the reduction of memory rather than FLOPs, and thus we employ the memory reduction of each channel to normalize the importance. Our method can be used to prune any structures including those with coupled channels. We conduct extensive experiments on various backbones, including the classic ResNet and ResNeXt, mobilefriendly MobileNetV2, and the NAS-based RegNet, both on image classification and object detection which is under-explored. Experimental results validate that our method can effectively prune sophisticated networks, boosting inference speed without sacrificing accuracy.

![pipeline](https://github.com/jshilong/FisherPruning/blob/main/resources/structures.png)

## Results and models

### Classification on ImageNet

| Model                    | Top-1 | Gap   | Flop(G) | Remained | Parameters(M) | Remained | Config       | Download                 |
| ------------------------ | ----- | ----- | ------- | -------- | ------------- | -------- | ------------ | ------------------------ |
| ResNet50                 | 76.55 | -     | 4.11    | -        | 25.6          | -        | [mmcls](<>)  | [model](<>) \| [log](<>) |
| ResNet50_pruned_act      | 75.22 | -1.33 | 2.06    | 50.1%    | 16.3          | 63.7%    | [config](<>) | [model](<>) \| [log](<>) |
| ResNet50_pruned_flops    | 75.61 | -0.94 | 2.06    | 50.1%    | 16.3          | 63.7%    | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2              | 71.86 | -     | 0.313   | -        | 3.51          | -        | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2_pruned_act   | 70.82 | -1.04 | 0.207   | 66.1%    | 3.18          | 90.6%    | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2_pruned_flops | 70.87 | -0.99 | 0.207   | 66.1%    | 2.82          | 88.7%    | [config](<>) | [model](<>) \| [log](<>) |

### Detection on COCO

| Model(Detector-Backbone)       | AP   | Gap | Flop(G) | Pruned | Parameters | Pruned | Config      | Download                 |
| ------------------------------ | ---- | --- | ------- | ------ | ---------- | ------ | ----------- | ------------------------ |
| RetinaNet-R50-FPN              | 36.5 | -   | 250     | -      | 63.8       | -      | [mmcls](<>) | [model](<>) \| [log](<>) |
| RetinaNet-R50-FPN_pruned_act   | 36.5 | -   | 126     | -      | 34.6       | -      | [mmcls](<>) | [model](<>) \| [log](<>) |
| RetinaNet-R50-FPN_pruned_flops | 36.6 | -   | 126     | -      | 34.9       | -      | [mmcls](<>) | [model](<>) \| [log](<>) |

## Get Started

### Pruning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  configs/pruning/path to config/group_fisher_xxx_model_name.py 8 \
  --work-dir $WORK_DIR
```

### Finetune

Update the `pruned_path` to your local file path that saves the pruned checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  configs/pruning/path to config/group_fisher_xxx_finetune_model_name.py.py 8 \
  --work-dir $WORK_DIR
```

### Deploy

TODO

## Citation

@InProceedings{liu2021group,
title = {Group Fisher Pruning for Practical Network Compression},
author =       {Liu, Liyang and Zhang, Shilong and Kuang, Zhanghui and Zhou, Aojun and Xue, Jing-Hao and Wang, Xinjiang and Chen, Yimin and Yang, Wenming and Liao, Qingmin and Zhang, Wayne},
booktitle = {Proceedings of the 38th International Conference on Machine Learning},
year = {2021},
series = {Proceedings of Machine Learning Research},
month = {18--24 Jul},
publisher ={PMLR},
}
