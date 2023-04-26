# DMCP: Differentiable Markov Channel Pruning for Neural Networks

## Abstract

Recent works imply that the channel pruning can be regarded as searching optimal sub-structure from unpruned networks. However, existing works based on this observation require training and evaluating a large number of structures, which limits their application. In this paper, we propose a novel differentiable method for channel pruning, named Differentiable Markov Channel Pruning (DMCP), to efficiently search the optimal sub-structure. Our method is differentiable and can be directly optimized by gradient descent with respect to standard task loss and budget regularization (e.g. FLOPs constraint). In DMCP, we model the channel pruning as a Markov process, in which each state represents for retaining the corresponding channel during pruning, and transitions between states denote the pruning process. In the end, our method is able to implicitly select the proper number of channels in each layer by the Markov process with optimized transitions. To validate the effectiveness of our method, we perform extensive experiments on Imagenet with ResNet and MobilenetV2. Results show our method can achieve consistent improvement than stateof-the-art pruning methods in various FLOPs settings.

## Getting Started

#### Train DMCP from scratch

```bash
GPUS=32 sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/pruning/mmcls/dmcp/dmcp_resnet50_supernet_32xb64.py \
  --work-dir $WORK_DIR
```

#### After the previous steps, retrain the selected pruned sub-network

#### with 2GFLOPs based on the output structure

#### 'DMCP_R50_2G.json'(SOURCECODE)

```bash
GPUS=32 sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/pruning/mmcls/dmcp/dmcp_resnet50_subnet_32xb64.py \
  --work-dir $WORK_DIR
```

<!-- ## Results and models

### 1.Classification

| Dataset  |  Supernet   |    Flops(M)    | Top-1 (%) | Top-5 (%) |                 config                 |                                                                                                                           Download                                                                                                                           |                                               Remark                                                |
| :------: | :---------: | :------------: | :-------: | :-------: | :------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| ImageNet | MobilenetV2 | 319M(Supernet) |   72.30   |   90.42   |                   -                    |                                                                                                                              -                                                                                                                               |                                                  -                                                  |
| ImageNet | MobilenetV2 |  209M(Subnet)  |   71.94   |   90.05   | [config](./dmcp_mbv2_subnet_32xb64.py) | [model](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/mobilenetv2/200M/DMCP_MBV2_200M.pth)  / [log](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/mobilenetv2/200M/dmcp_mobilenetv2_supernet_32xb64_target_flops_200m_20230129_184919.log) | [arch](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/mobilenetv2/200M/DMCP_MBV2_200M.json) |
| ImageNet | MobilenetV2 |  102M(Subnet)  |   67.22   |   88.61   | [config](./dmcp_mbv2_subnet_32xb64.py) | [model](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/mobilenetv2/100M/DMCP_MBV2_100M.pth) / [log](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/mobilenetv2/100M/dmcp_mobilenetv2_supernet_32xb64_target_flops_100m_20230129_184919.log)  |                                   [arch\*](./DMCP_MBV2_100M.json)                                   | -->

<!-- The result of Resnet50 has bugs. -->

<!-- | ImageNet |  ResNet50   | 4.09G(Supernet) |   77.46   |   93.55   |                     -                      |                                                                                                                              -                                                                                                                               |                                                  -                                                  |
| ImageNet |  ResNet50   |  2.07G(Subnet)  |   76.11   |   93.01   | [config](./dmcp_resnet50_subnet_32xb64.py) |          [model](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/resnet50/2G/DMCP_R50_2G.pth)  / [log](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/resnet50/2G/dmcp_resnet50_supernet_32xb64_target_flops_2g_20230129_112944.log)          |                                    [arch\*](./DMCP_R50_2G.json)                                     |
| ImageNet |  ResNet50   |  1.05G(Subnet)  |   74.12   |   92.33   | [config](./dmcp_resnet50_subnet_32xb64.py) |          [model](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/resnet50/1G/DMCP_R50_1G.pth) / [log](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/resnet50/1G/dmcp_resnet50_supernet_32xb64_target_flops_1g_20230107_223552.log)           |     [arch](https://download.openmmlab.com/mmrazor/v1/pruning/dmcp/resnet50/1G/DMCP_R50_1G.json)     | -->

<!-- **Note**

1. Arch with * are converted from the [official repo](https://github.com/Zx55/dmcp).
2. To get the sub-network structure with different pruning rates, we support modifying `target_flops` in `model` in the supernet config, note that here it is in MFLOPs. For example, `target_flops=1000` means get subnet with 1GFLOPs.
3. When outputting the sampled sub-network, the FLOPs will fluctuate around 5% around the target value for efficiency.
4. More models with different pruning rates will be released later. -->

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
