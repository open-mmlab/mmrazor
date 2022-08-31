# BYOT

> [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/abs/1905.08094)

<!-- [ALGORITHM] -->

## Abstract

Convolutional neural networks have been widely deployed in various application scenarios. In order to extend the applications' boundaries to some accuracy-crucial domains, researchers have been investigating approaches to boost accuracy through either deeper or wider network structures, which brings with them the exponential increment of the computational and storage cost, delaying the responding time. In this paper, we propose a general training framework named self distillation, which notably enhances the performance (accuracy) of convolutional neural networks through shrinking the size of the network rather than aggrandizing it. Different from traditional knowledge distillation - a knowledge transformation methodology among networks, which forces student neural networks to approximate the softmax layer outputs of pre-trained teacher neural networks, the proposed self distillation framework distills knowledge within network itself. The networks are firstly divided into several sections. Then the knowledge in the deeper portion of the networks is squeezed into the shallow ones. Experiments further prove the generalization of the proposed self distillation framework: enhancement of accuracy at average level is 2.65%, varying from 0.61% in ResNeXt as minimum to 4.07% in VGG19 as maximum. In addition, it can also provide flexibility of depth-wise scalable inference on resource-limited edge devices.Our codes will be released on github soon. [Unofficial code](https://github.com/luanyunteng/pytorch-be-your-own-teacher)

## Pipeline

![byot](https://user-images.githubusercontent.com/88702197/187422992-e7bd692d-b6d4-44d8-8b36-741e0cf1c4f6.png)

## Results and models

#### Classification

| Location | Dataset  |                     Model                     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                                                                               Download                                                                                                                |
| :------: | :------: | :-------------------------------------------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  logits  | CIFAR100 | [R18_BYOT](./byot_resnet18_8xb16_cifar100.py) |   11.22   |   0.56   |   80.66   |   95.76   | [model](https://download.openmmlab.com/mmrazor/v1/byot/byot_resnet18_8xb16_cifar100_20220817_191217-0251084e.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/byot/byot_resnet18_8xb16_cifar100_20220817_191217-0251084e.json) |

## Citation

```latex
@ARTICLE{2019arXiv190508094Z,
       author = {{Zhang}, Linfeng and {Song}, Jiebo and {Gao}, Anni and {Chen}, Jingwei and {Bao}, Chenglong and {Ma}, Kaisheng},
        title = {Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation},
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = 2019,
        month = may,
          eid = {arXiv:1905.08094},
        pages = {arXiv:1905.08094},
archivePrefix = {arXiv},
       eprint = {1905.08094},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190508094Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Get Started

### Distillation training.

```bash
sh tools/dist_train.sh \
  configs/distill/mmcls/byot/byot_logits_resnet18_cifar100_8xb16_in1k.py 8
```

### Test

```bash
sh tools/dist_train.sh \
  configs/distill/mmcls/byot/byot_logits_resnet18_cifar100_8xb16_in1k.py 8
```
