# MGD

> [Masked Generative Distillation](https://arxiv.org/abs/2205.01529)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation has been applied to various tasks successfully. The current distillation algorithm usually improves students' performance by imitating the output of the teacher. This paper shows that teachers can also improve students' representation power by guiding students' feature recovery. From this point of view, we propose Masked Generative Distillation (MGD), which is simple: we mask random pixels of the student's feature and force it to generate the teacher's full feature through a simple block. MGD is a truly general feature-based distillation method, which can be utilized on various tasks, including image classification, object detection, semantic segmentation and instance segmentation. We experiment on different models with extensive datasets and the results show that all the students achieve excellent improvements. Notably, we boost ResNet-18 from 69.90% to 71.69% ImageNet top-1 accuracy, RetinaNet with ResNet-50 backbone from 37.4 to 41.0 Boundingbox mAP, SOLO based on ResNet-50 from 33.1 to 36.2 Mask mAP and DeepLabV3 based on ResNet-18 from 73.20 to 76.02 mIoU.

![pipeline](https://github.com/yzd-v/MGD/raw/master/architecture.png)

## Results and models

### Detection

| Location | Dataset |                                                            Teacher                                                             |                                                        Student                                                         | Lr schd | mAP  | mAP(T) | mAP(S) |                       Config                        | Download                                                                                                                                                                                                                                                                                                                                                                                                              |
| :------: | :-----: | :----------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :-----: | :--: | :----: | :----: | :-------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   FPN    |  COCO   | [RetinaNet-X101](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py) | [RetinaNet-R50](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/retinanet/retinanet_r50_fpn_2x_coco.py) |   2x    | 41.0 |  41.0  |  37.4  | [config](mgd_fpn_retina_x101_retina_r50_2x_coco.py) | [teacher](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/mgd/mgd_fpn_retina_x101_retina_r50_2x_coco_20221209_191847-87141529.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/mgd/mgd_fpn_retina_x101_retina_r50_2x_coco_20221209_191847-87141529.log) |

## Citation

```latex
@article{yang2022masked,
  title={Masked Generative Distillation},
  author={Yang, Zhendong and Li, Zhe and Shao, Mingqi and Shi, Dachuan and Yuan, Zehuan and Yuan, Chun},
  journal={arXiv preprint arXiv:2205.01529},
  year={2022}
}
```
