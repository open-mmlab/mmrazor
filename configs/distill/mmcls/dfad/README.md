# Data-Free Adversarial Distillation (DFAD)

> [Data-Free Adversarial Distillation](https://arxiv.org/pdf/1912.11006.pdf)

<!-- [ALGORITHM] -->

## Abstract

Knowledge Distillation (KD) has made remarkable progress in the last few years and become a popular paradigm for model compression and knowledge transfer. However, almost all existing KD algorithms are data-driven, i.e., relying on a large amount of original training data or alternative data, which is usually unavailable in real-world scenarios. In this paper, we devote ourselves to this challenging problem and propose a novel adversarial distillation mechanism to craft a compact student model without any real-world data. We introduce a model discrepancy to quantificationally measure the difference between student and teacher models and construct an optimizable upper bound. In our work, the student and the teacher jointly act the role of the discriminator to reduce this discrepancy, when a generator adversarially produces some "hard samples" to enlarge it. Extensive experiments demonstrate that the proposed data-free method yields comparable performance to existing data-driven methods. More strikingly, our approach can be directly extended to semantic segmentation, which is more complicated than classification, and our approach achieves state-of-the-art results.

<img width="1001" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187423332-30a5d409-6f83-45d7-9e11-e306f7ffec78.png">

## Results and models

### Classification

| Location | Dataset |                                                     Teacher                                                     |                                                     Student                                                     |  Acc  | Acc(T) | Acc(S) |                           Config                           | Download                                                                                                                                                                                                                                                                                                                                                                                |
| :------: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :--------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  logits  | Cifar10 | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb16_cifar10.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | 92.80 | 95.34  | 94.82  | [config](./dfad_logits_resnet34_resnet18_8xb32_cifar10.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/DFAD/dfad_logits_resnet34_resnet18_8xb32_cifar10_20220819_051141-961a5b09.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/DFAD/dfad_logits_resnet34_resnet18_8xb32_cifar10_20220819_051141-961a5b09.json) |

## Citation

```latex
@article{fang2019data,
  title={Data-free adversarial distillation},
  author={Fang, Gongfan and Song, Jie and Shen, Chengchao and Wang, Xinchao and Chen, Da and Song, Mingli},
  journal={arXiv preprint arXiv:1912.11006},
  year={2019}
}
```
