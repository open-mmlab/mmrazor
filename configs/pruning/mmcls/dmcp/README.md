# DMCP: Differentiable Markov Channel Pruning for Neural Networks

## Abstract

Recent works imply that the channel pruning can be regarded as searching optimal sub-structure from unpruned networks. However, existing works based on this observation require training and evaluating a large number of structures, which limits their application. In this paper, we propose a novel differentiable method for channel pruning, named Differentiable Markov Channel Pruning (DMCP), to efficiently search the optimal sub-structure. Our method is differentiable and can be directly optimized by gradient descent with respect to standard task loss and budget regularization (e.g. FLOPs constraint). In DMCP, we model the channel pruning as a Markov process, in which each state represents for retaining the corresponding channel during pruning, and transitions between states denote the pruning process. In the end, our method is able to implicitly select the proper number of channels in each layer by the Markov process with optimized transitions. To validate the effectiveness of our method, we perform extensive experiments on Imagenet with ResNet and MobilenetV2. Results show our method can achieve consistent improvement than stateof-the-art pruning methods in various FLOPs settings.

## Citation

```latex
@inproceedings{guo2020dmcp,
  title={Dmcp: Differentiable markov channel pruning for neural networks},
  author={Guo, Shaopeng and Wang, Yujie and Li, Quanquan and Yan, Junjie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1539--1547},
  year={2020}
}
```

## Results and models

### 1.Classification

| Dataset  | Supernet | Flops(M) | Top-1 (%) | Top-5 (%) |                   Config                    |                         Download                         |
| :------: | :------: | :------: | :-------: | :-------: | :-----------------------------------------: | :------------------------------------------------------: |
| ImageNet | ResNet50 |    -     |     -     |     -     | [config](./dmcp_resnet50_supernet_8xb32.py) | \[model\] / [arch](./DMCP_SUBNET_IMAGENET.yaml)/ \[log\] |

## Getting Started

#### Train DMCP from scratch

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/pruning/mmcls/dmcp/dmcp_resnet50_supernet_8xb32.py \
  --work-dir $WORK_DIR
```

#### After the previous steps, retrain the selected sub-network based on

#### the output structure 'DMCP_SUBNET_IMAGENET.yaml'

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/pruning/mmcls/dmcp/dmcp_resnet50_subnet_8xb32.py \
  --work-dir $WORK_DIR
```
