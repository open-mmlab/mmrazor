# FGD

> [Focal and Global Knowledge Distillation for Detectors](https://arxiv.org/abs/2111.11837)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation has been applied to image classification successfully. However, object detection is much more sophisticated and most knowledge distillation methods have failed on it. In this paper, we point out that in object detection, the features of the teacher and student vary greatly in different areas, especially in the foreground and background. If we distill them equally, the uneven differences between feature maps will negatively affect the distillation. Thus, we propose Focal and Global Distillation (FGD). Focal distillation separates the foreground and background, forcing the student to focus on the teacher's critical pixels and channels. Global distillation rebuilds the relation between different pixels and transfers it from teachers to students, compensating for missing global information in focal distillation. As our method only needs to calculate the loss on the feature map, FGD can be applied to various detectors. We experiment on various detectors with different backbones and the results show that the student detector achieves excellent mAP improvement. For example, ResNet-50 based RetinaNet, Faster RCNN, RepPoints and Mask RCNN with our distillation method achieve 40.7%, 42.0%, 42.0% and 42.1% mAP on COCO2017, which are 3.3, 3.6, 3.4 and 2.9 higher than the baseline, respectively.

![pipeline](https://user-images.githubusercontent.com/41630003/220037957-25a1440f-fcb3-413a-a350-97937bf6a042.png)

## Results and models

### Detection

| Location | Dataset |                                                            Teacher                                                            |                                                        Student                                                        | mAP  | mAP(T) | mAP(S) |                          Config                           | Download                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :------: | :-----: | :---------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :--: | :----: | :----: | :-------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   FPN    |  COCO   | [retina_x101_1x](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py) | [retina_r50_2x](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | 40.5 |  41.0  |  37.4  | [config](./fgd_retina_x101_fpn_retina_r50_fpn_2x_coco.py) | [teacher](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/distill/fgd/fgd_retina_x101_retina_r50_2x_coco_20221216_114845-c4c7496d.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/distill/fgd/fgd_retina_x101_retina_r50_2x_coco_20221216_114845-c4c7496d.json) |

## Citation

```latex
@article{yang2021focal,
  title={Focal and Global Knowledge Distillation for Detectors},
  author={Yang, Zhendong and Li, Zhe and Jiang, Xiaohu and Gong, Yuan and Yuan, Zehuan and Zhao, Danpei and Yuan, Chun},
  journal={arXiv preprint arXiv:2111.11837},
  year={2021}
}
```
