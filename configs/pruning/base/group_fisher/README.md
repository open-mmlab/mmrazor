# Group_fisher pruning

> [Group Fisher Pruning for Practical Network Compression.](https://arxiv.org/pdf/2108.00708.pdf)

## Abstract

Network compression has been widely studied since it is able to reduce the memory and computation cost during inference. However, previous methods seldom deal with complicated structures like residual connections, group/depthwise convolution and feature pyramid network, where channels of multiple layers are coupled and need to be pruned simultaneously. In this paper, we present a general channel pruning approach that can be applied to various complicated structures. Particularly, we propose a layer grouping algorithm to find coupled channels automatically. Then we derive a unified metric based on Fisher information to evaluate the importance of a single channel and coupled channels. Moreover, we find that inference speedup on GPUs is more correlated with the reduction of memory rather than FLOPs, and thus we employ the memory reduction of each channel to normalize the importance. Our method can be used to prune any structures including those with coupled channels. We conduct extensive experiments on various backbones, including the classic ResNet and ResNeXt, mobilefriendly MobileNetV2, and the NAS-based RegNet, both on image classification and object detection which is under-explored. Experimental results validate that our method can effectively prune sophisticated networks, boosting inference speed without sacrificing accuracy.

![pipeline](https://github.com/jshilong/FisherPruning/blob/main/resources/structures.png?raw=true)

## Results and models

### Classification on ImageNet

| Model                    | Top-1 | Gap   | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config       | Download                 |
| ------------------------ | ----- | ----- | ------- | --------- | ------------- | --------- | ------------ | ------------------------ |
| ResNet50                 | 76.55 | -     | 4.11    | -         | 25.6          | -         | [mmcls](<>)  | [model](<>) \| [log](<>) |
| ResNet50_pruned_act      | 75.22 | -1.33 | 2.06    | 50.1%     | 16.3          | 63.7%     | [config](<>) | [model](<>) \| [log](<>) |
| ResNet50_pruned_flops    | 75.61 | -0.94 | 2.06    | 50.1%     | 16.3          | 63.7%     | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2              | 71.86 | -     | 0.313   | -         | 3.51          | -         | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2_pruned_act   | 70.82 | -1.04 | 0.207   | 66.1%     | 3.18          | 90.6%     | [config](<>) | [model](<>) \| [log](<>) |
| MobileNetV2_pruned_flops | 70.87 | -0.99 | 0.207   | 66.1%     | 2.82          | 88.7%     | [config](<>) | [model](<>) \| [log](<>) |

### Detection on COCO

| Model(Detector-Backbone)       | AP   | Gap | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config      | Download                 |
| ------------------------------ | ---- | --- | ------- | --------- | ------------- | --------- | ----------- | ------------------------ |
| RetinaNet-R50-FPN              | 36.5 | -   | 250     | -         | 63.8          | -         | [mmcls](<>) | [model](<>) \| [log](<>) |
| RetinaNet-R50-FPN_pruned_act   | 36.5 | -   | 126     | 50.4%     | 34.6          | 54.2%     | [mmcls](<>) | [model](<>) \| [log](<>) |
| RetinaNet-R50-FPN_pruned_flops | 36.6 | -   | 126     | 50.4%     | 34.9          | 54.7%     | [mmcls](<>) | [model](<>) \| [log](<>) |

## Get Started

We have three steps to apply GroupFisher to your model, including Prune, Finetune, Deploy.

### Prune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  {config_folder}/group_fisher_{normalization_type}_prune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

In the pruning config file. You have to fill some args as below.

```python
"""
_base_ (str): The path to your pretrained model checkpoint.
pretrained_path (str): The path to your pretrained model checkpoint.

interval (int): Interval between pruning two channels. You should ensure you
    can reach your target pruning ratio when the training ends.
normalization_type (str): GroupFisher uses two methods to normlized the channel
    importance, including ['flops','act']. The former uses flops, while the
    latter uses the memory occupation of activation feature maps.
lr_ratio (float): Ratio to decrease lr rate. As pruning progress is unstable,
    you need to decrease the original lr rate until the pruning training work
    steadly without getting nan.

target_flop_ratio (float): The target flop ratio to prune your model.
input_shape (Tuple): input shape to measure the flops.
"""
```

After the pruning process, you will get a checkpoint of the pruned model named flops\_{target_flop_ratio}.pth in your workdir.

### Finetune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

There are also some args for you to fill in the config file as below.

```python
"""
_base_ (str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
"""
```

After finetuning, except a checkpoint of the best model, there is also a fix_subnet.json, which records the pruned model structure. It will be used when deploying.

### Deploy

First, we assume you are fimilar to mmdeploy. For a pruned model, you only need to use the pruning deploy config to instead the pretrain config to deploy the pruned version of your model.

```bash
python {mmdeploy}/tools/deploy.py \
    {mmdeploy}/configs/mmcls/classification_onnxruntime_dynamic.py \
    {config_folder}/group_fisher_{normalization_type}_deploy_{model_name}.py \
    {path_to_finetuned_checkpoint}.pth \
    {mmdeploy}/tests/data/tiger.jpeg
```

The deploy config has some args as below:

```python
"""
_base_ (str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""
```

The divisor is important for the actual inference speed, and we suggest you to test it in \[1,2,4,8,16,32\] to find the fastest divisor.

## Implementation

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
