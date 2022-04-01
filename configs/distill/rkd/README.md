# RKD



> [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)

<!-- [ALGORITHM] -->
## Abstract
Knowledge distillation aims at transferring knowledge acquired
in one model (a teacher) to another model (a student) that is
typically smaller. Previous approaches can be expressed as
a form of training the student to mimic output activations of
individual data examples represented by the teacher. We introduce
a novel approach, dubbed relational knowledge distillation (RKD),
that transfers mutual relations of data examples instead.
For concrete realizations of RKD, we propose distance-wise and
angle-wise distillation losses that penalize structural differences
in relations. Experiments conducted on different tasks show that the
proposed method improves educated student models with a significant margin.
In particular for metric learning, it allows students to outperform their
teachers' performance, achieving the state of the arts on standard benchmark datasets.

![pipeline](/docs/en/imgs/model_zoo/rkd/pipeline.png)

## Results and models
### Classification
|Location|Dataset|Teacher|Student|Acc|Acc(T)|Acc(S)|Config | Download |
:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:---------|
| neck     |ImageNet|[resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py)|[resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py)| 70.23 |    73.62 |    69.90  |[config](./rkd_neck_resnet34_resnet18_8xb32_in1k.py)|[teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) &#124;[model](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet34_resnet18_8xb32_in1k_acc-70.23_20220401-f25700ac.pth) &#124; [log](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet34_resnet18_8xb32_in1k_20220312_130419.log.json)|
| neck     |Cifar100|[resnet50](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb16_cifar100.py)|[vgg11bn]()| 73.11 |    79.90 |    71.26  |[config](./rkd_neck_resnet50_vgg11bn_8xb16_cifar100.py)|[teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth) &#124;[model](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet50_vgg11bn_8xb16_cifar100_acc-73.11_20220401-5422b329.pth) &#124; [log](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet50_vgg11bn_8xb16_cifar100_20220401_124319.log.json)|



## Citation
```latex
@inproceedings{park2019relational,
  title={Relational knowledge distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
