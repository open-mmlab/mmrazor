# Paraphrasing Complex Network: Network Compression via Factor Transfer

## Abstract

Many researchers have sought ways of model compression to reduce the size of a deep neural network (DNN) with minimal performance degradation in order to use DNNs in embedded systems. Among the model compression methods, a method called knowledge transfer is to train a student network with a stronger teacher network. In this paper, we propose a novel knowledge transfer method which uses convolutional operations to paraphrase teacher’s knowledge and to translate it for the student. This is done by two convolutional modules, which are called a paraphraser and a translator. The paraphraser is trained in an unsupervised manner to extract the teacher factors which are defined as paraphrased information of the teacher network. The translator located at the student network extracts the student factors and helps to translate the teacher factors by mimicking them. We observed that our student network trained with the proposed factor transfer method outperforms the ones trained with conventional knowledge transfer methods. The original code is available at this [link](https://github.com/Jangho-Kim/Factor-Transfer-pytorch)

## Results and models

| Dataset | Model     | Teacher   | Top-1 (%) | Top-5 (%) | Configs                                                                                                                                                            | Download          |
| ------- | --------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| CIFAR10 | ResNet-18 | ResNet-50 | 94.86     | 99.88     | [pretrain](./factor_transfer_backbone_resnet50_resnet18_8xb32_cifar10_pretrain.py) \| [train](./factor_transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py) | [model & log](<>) |

## Getting Started

### Connectors pre-training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor_transfer_backbone_resnet50_resnet18_8xb32_cifar10_pretrain.py \
  $PRETRAIN_WORK_DIR
```

### Distillation training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor_transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py \
  $DISTILLATION_WORK_DIR
```

### Test

```bash
sh tools/slurm_test.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/factor_transfer/factor_transfer_backbone_resnet50_resnet18_8xb32_cifar10_train.py \
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
