# Zero-shot Knowledge Transfer via Adversarial Belief Matching (ZSKT)

> [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)

<!-- [ALGORITHM] -->

## Abstract

Performing knowledge transfer from a large teacher network to a smaller student is a popular task in modern deep learning applications. However, due to growing dataset sizes and stricter privacy regulations, it is increasingly common not to have access to the data that was used to train the teacher. We propose a novel method which trains a student to match the predictions of its teacher without using any data or metadata. We achieve this by training an adversarial generator to search for images on which the student poorly matches the teacher, and then using them to train the student. Our resulting student closely approximates its teacher for simple datasets like SVHN, and on CIFAR10 we improve on the state-of-the-art for few-shot distillation (with 100 images per class), despite using no data. Finally, we also propose a metric to quantify the degree of belief matching between teacher and student in the vicinity of decision boundaries, and observe a significantly higher match between our zero-shot student and the teacher, than between a student distilled with real data and the teacher. Code available at: https://github.com/polo5/ZeroShotKnowledgeTransfer

## The teacher and student decision boundaries

<img width="766" alt="distribution" src="https://user-images.githubusercontent.com/88702197/187424317-9f3c5547-a838-4858-b63e-608eee8165f5.png">

## Pseudo images sampled from the generator

<img width="1176" alt="synthesis" src="https://user-images.githubusercontent.com/88702197/187424322-79be0b07-66b5-4775-8e23-6c2ddca0ad0f.png">

## Results and models

### Classification

|     Location      | Dataset |                                                     Teacher                                                     |                                                     Student                                                     |  Acc  | Acc(T) | Acc(S) |                               Config                                | Download                                                                                                                                                                                                                                                                                                                                                                                                  |
| :---------------: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backbone & logits | Cifar10 | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb16_cifar10.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | 93.05 | 95.34  | 94.82  | [config](./zskt_backbone_logits_resnet34_resnet18_8xb16_cifar10.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/ZSKT/zskt_backbone_logits_resnet34_resnet18_8xb16_cifar10_20220823_114006-28584c2e.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/ZSKT/zskt_backbone_logits_resnet34_resnet18_8xb16_cifar10_20220823_114006-28584c2e.json) |

## Citation

```latex
@article{micaelli2019zero,
  title={Zero-shot knowledge transfer via adversarial belief matching},
  author={Micaelli, Paul and Storkey, Amos J},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```
