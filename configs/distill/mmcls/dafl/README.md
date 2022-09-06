# Data-Free Learning of Student Networks (DAFL)

> [Data-Free Learning of Student Networks](https://doi.org/10.1109/ICCV.2019.00361)

<!-- [ALGORITHM] -->

## Abstract

Learning portable neural networks is very essential for computer vision for the purpose that pre-trained heavy deep models can be well applied on edge devices such as mobile phones and micro sensors. Most existing deep neural network compression and speed-up methods are very effective for training compact deep models, when we can directly access the training dataset. However, training data for the given deep network are often unavailable due to some practice problems (e.g. privacy, legal issue, and transmission), and the architecture of the given network are also unknown except some interfaces. To this end, we propose a novel framework for training efficient deep neural networks by exploiting generative adversarial networks (GANs). To be specific, the pre-trained teacher networks are regarded as a fixed discriminator and the generator is utilized for deviating training samples which can obtain the maximum response on the discriminator. Then, an efficient network with smaller model size and computational complexity is trained using the generated data and the teacher network, simultaneously. Efficient student networks learned using the pro- posed Data-Free Learning (DAFL) method achieve 92.22% and 74.47% accuracies using ResNet-18 without any training data on the CIFAR-10 and CIFAR-100 datasets, respectively. Meanwhile, our student network obtains an 80.56% accuracy on the CelebA benchmark.

<img width="910" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187423163-b34896fc-8516-403b-acd7-4c0b8e43af5b.png">

## Results and models

### Classification

|     Location      | Dataset |                                                     Teacher                                                     |                                                     Student                                                     |  Acc  | Acc(T) | Acc(S) |                           Config                            | Download                                                                                                                                                                                                                                                                                                                                                                                  |
| :---------------: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :---------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backbone & logits | Cifar10 | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb16_cifar10.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | 93.27 | 95.34  | 94.82  | [config](./dafl_logits_resnet34_resnet18_8xb256_cifar10.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/DAFL/dafl_logits_resnet34_resnet18_8xb256_cifar10_20220815_202654-67142167.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/DAFL/dafl_logits_resnet34_resnet18_8xb256_cifar10_20220815_202654-67142167.json) |

## Citation

```latex
@inproceedings{DBLP:conf/iccv/ChenW0YLSXX019,
  author    = {Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu,
               Boxin Shi, Chunjing Xu, Chao Xu and Qi Tian},
  title     = {Data-Free Learning of Student Networks},
  booktitle = {2019 {IEEE/CVF} International Conference on Computer Vision, {ICCV}
               2019, Seoul, Korea (South), October 27 - November 2, 2019},
  pages     = {3513--3521},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICCV.2019.00361},
  doi       = {10.1109/ICCV.2019.00361},
  timestamp = {Mon, 17 May 2021 08:18:18 +0200},
  biburl    = {https://dblp.org/rec/conf/iccv/ChenW0YLSXX019.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
```
