# Zero-shot Knowledge Transfer via Adversarial Belief Matching (ZSKT)

> [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)

<!-- [ALGORITHM] -->

## Abstract

Performing knowledge transfer from a large teacher network to a smaller student is a popular task in modern deep learning applications. However, due to growing dataset sizes and stricter privacy regulations, it is increasingly common not to have access to the data that was used to train the teacher. We propose a novel method which trains a student to match the predictions of its teacher without using any data or metadata. We achieve this by training an adversarial generator to search for images on which the student poorly matches the teacher, and then using them to train the student. Our resulting student closely approximates its teacher for simple datasets like SVHN, and on CIFAR10 we improve on the state-of-the-art for few-shot distillation (with 100 images per class), despite using no data. Finally, we also propose a metric to quantify the degree of belief matching between teacher and student in the vicinity of decision boundaries, and observe a significantly higher match between our zero-shot student and the teacher, than between a student distilled with real data and the teacher. Code available at: https://github.com/polo5/ZeroShotKnowledgeTransfer

## The teacher and student decision boundaries

![ZSKT_Distribution](/docs/en/imgs/model_zoo/zskt/zskt_distribution.png)

## Pseudo images sampled from the generator

![ZSKT_Fakeimgs](/docs/en/imgs/model_zoo/zskt/zskt_synthesis.png)

## Results and models

### Classification

|     Location      | Dataset |                                                     Teacher                                                     |                                                     Student                                                     |  Acc  | Acc(T) | Acc(S) |                      Config                       | Download                                                                                                                                     |
| :---------------: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-----------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------- |
| backbone & logits | Cifar10 | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb16_cifar10.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | 93.50 | 95.34  | 94.82  | [config](./dafl_logits_r34_r18_8xb256_cifar10.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \|[model](<>) \| [log](<>) |

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

## Acknowledgement

Appreciate Davidgzx's contribution.
