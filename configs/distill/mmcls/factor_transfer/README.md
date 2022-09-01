# Paraphrasing Complex Network: Network Compression via Factor Transfer

> [Paraphrasing Complex Network: Network Compression via Factor Transfer](https://arxiv.org/abs/1802.04977)

<!-- [ALGORITHM] -->

## Abstract

Many researchers have sought ways of model compression to reduce the size of a deep neural network (DNN) with minimal performance degradation in order to use DNNs in embedded systems. Among the model compression methods, a method called knowledge transfer is to train a student network with a stronger teacher network. In this paper, we propose a novel knowledge transfer method which uses convolutional operations to paraphrase teacher’s knowledge and to translate it for the student. This is done by two convolutional modules, which are called a paraphraser and a translator. The paraphraser is trained in an unsupervised manner to extract the teacher factors which are defined as paraphrased information of the teacher network. The translator located at the student network extracts the student factors and helps to translate the teacher factors by mimicking them. We observed that our student network trained with the proposed factor transfer method outperforms the ones trained with conventional knowledge transfer methods. The original code is available at this [link](https://github.com/Jangho-Kim/Factor-Transfer-pytorch)

## Results and models

| Dataset | Model     | Teacher   | Top-1 (%) | Top-5 (%) | Configs                                                                                                                                                            | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------- | --------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CIFAR10 | ResNet-18 | ResNet-50 | 94.86     | 99.88     | [pretrain](./factor-transfer_backbone_resnet50_resnet18_8xb32_cifar10_pretrain.py) \| [train](./factor-transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py) | [pretrain model](https://download.openmmlab.com/mmrazor/v1/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_pretrain_20220831_173259-ebdb09e2.pth) \| [pretrain log](https://download.openmmlab.com/mmrazor/v1/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_pretrain_20220831_173259-ebdb09e2.json) \| [train model](https://download.openmmlab.com/mmrazor/v1/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_train_20220831_201322-943df33f.pth) \| [train log](https://download.openmmlab.com/mmrazor/v1/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_train_20220831_201322-943df33f.json) |

## Getting Started

### Connectors pre-training.

```bash
sh tools/dist_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb32_cifar10_pretrain.py \
  $PRETRAIN_WORK_DIR
```

### Distillation training.

```bash
sh tools/dist_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py \
  $DISTILLATION_WORK_DIR
```

### Test

```bash
sh tools/dist_test.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor-transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py \
  $DISTILLATION_WORK_DIR/latest.sh --eval $EVAL_SETTING
```

## Citation

```latex
@inproceedings{kim2018paraphrasing,
  title={Paraphrasing complex network: network compression via factor transfer},
  author={Kim, Jangho and Park, SeongUk and Kwak, Nojun},
  booktitle={Proceedings of the 32nd International Conference on Neural Information Processing Systems},
  pages={2765--2774},
  year={2018}
}
```
