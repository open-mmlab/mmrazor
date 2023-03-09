# Group_fisher pruning

> [Group Fisher Pruning for Practical Network Compression.](https://arxiv.org/pdf/2108.00708.pdf)

## Abstract

Network compression has been widely studied since it is able to reduce the memory and computation cost during inference. However, previous methods seldom deal with complicated structures like residual connections, group/depthwise convolution and feature pyramid network, where channels of multiple layers are coupled and need to be pruned simultaneously. In this paper, we present a general channel pruning approach that can be applied to various complicated structures. Particularly, we propose a layer grouping algorithm to find coupled channels automatically. Then we derive a unified metric based on Fisher information to evaluate the importance of a single channel and coupled channels. Moreover, we find that inference speedup on GPUs is more correlated with the reduction of memory rather than FLOPs, and thus we employ the memory reduction of each channel to normalize the importance. Our method can be used to prune any structures including those with coupled channels. We conduct extensive experiments on various backbones, including the classic ResNet and ResNeXt, mobilefriendly MobileNetV2, and the NAS-based RegNet, both on image classification and object detection which is under-explored. Experimental results validate that our method can effectively prune sophisticated networks, boosting inference speed without sacrificing accuracy.

![pipeline](https://github.com/jshilong/FisherPruning/blob/main/resources/structures.png?raw=true)

## Results and models

### Classification on ImageNet

| Model                         | Top-1 | Gap   | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config                                   | Download                                                    | Onnx_cpu(FPS) |
| ----------------------------- | ----- | ----- | ------- | --------- | ------------- | --------- | ---------------------------------------- | ----------------------------------------------------------- | ------------- |
| ResNet50                      | 76.55 | -     | 4.11    | -         | 25.6          | -         | [mmcls][cls_r50_c]                       | [model][cls_r50_m]                                          | 55.360        |
| ResNet50_pruned_act           | 75.22 | -1.33 | 2.06    | 50.1%     | 16.3          | 63.7%     | [prune][r_a_pc] \| [finetune][r_a_fc]    | [pruned][r_a_p] \| [finetuned][r_a_f] \| [log][r_a_l]       | 80.671        |
| ResNet50_pruned_act + dist kd | 76.50 | -0.05 | 2.06    | 50.1%     | 16.3          | 63.7%     | [prune][r_a_pc] \| [finetune][r_a_fc_kd] | [pruned][r_a_p] \| [finetuned][r_a_f_kd] \| [log][r_a_l_kd] | 80.671        |
| ResNet50_pruned_flops         | 75.61 | -0.94 | 2.06    | 50.1%     | 16.3          | 63.7%     | [prune][r_f_pc] \| [finetune][r_f_fc]    | [pruned][r_f_p] \| [finetuned][r_f_f] \| [log][r_f_l]       | 78.674        |
| MobileNetV2                   | 71.86 | -     | 0.313   | -         | 3.51          | -         | [mmcls][cls_m_c]                         | [model][cls_m_m]                                            | 419.673       |
| MobileNetV2_pruned_act        | 70.82 | -1.04 | 0.207   | 66.1%     | 3.18          | 90.6%     | [prune][m_a_pc] \| [finetune][m_a_fc]    | [pruned][m_a_p] \| [finetuned][m_a_f] \| [log][m_a_l]       | 576.118       |
| MobileNetV2_pruned_flops      | 70.87 | -0.99 | 0.207   | 66.1%     | 2.82          | 88.7%     | [prune][m_f_pc] \| [finetune][m_f_fc]    | [pruned][m_f_p] \| [finetuned][m_f_f] \| [log][m_f_l]       | 540.105       |

**Note**

- Because the pruning papers use different pretraining and finetuning settings, It is hard to compare them fairly. As a result, we prefer to apply algorithms on the openmmlab settings.
- This may make the experiment results are different from that in the original papers.

### Detection on COCO

| Model(Detector-Backbone)       | AP   | Gap  | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config                                  | Download                                                 | Onnx_cpu(FPS) |
| ------------------------------ | ---- | ---- | ------- | --------- | ------------- | --------- | --------------------------------------- | -------------------------------------------------------- | ------------- |
| RetinaNet-R50-FPN              | 36.5 | -    | 250     | -         | 63.8          | -         | [mmdet][det_rt_c]                       | [model][det_rt_m]                                        | 1.095         |
| RetinaNet-R50-FPN_pruned_act   | 36.5 | 0.0  | 126     | 50.4%     | 34.6          | 54.2%     | [prune][rt_a_pc] \| [finetune][rt_a_fc] | [pruned][rt_a_p] \| [finetuned][rt_a_f] \| [log][rt_a_l] | 1.608         |
| RetinaNet-R50-FPN_pruned_flops | 36.6 | +0.1 | 126     | 50.4%     | 34.9          | 54.7%     | [prune][rt_f_pc] \| [finetune][rt_f_fc] | [pruned][rt_f_p] \| [finetuned][rt_f_f] \| [log][rt_f_l] | 1.609         |

### Pose on COCO

| Model                | AP    | Gap    | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config                                  | Download                                                    | Onnx_cpu(FPS) |
| -------------------- | ----- | ------ | ------- | --------- | ------------- | --------- | --------------------------------------- | ----------------------------------------------------------- | ------------- |
| rtmpose-s            | 0.716 | -      | 0.68    | -         | 5.47          | -         | [mmpose][pose_s_c]                      | [model][pose_s_m]                                           | 196           |
| rtmpose-s_pruned_act | 0.691 | -0.025 | 0.34    | 50.0%     | 3.42          | 62.5%     | [prune][rp_a_pc] \| [finetune][rp_a_fc] | [pruned][rp_sc_p] \| [finetuned][rp_sc_f] \| [log][rp_sc_l] | 268           |
| rtmpose-t            | 0.682 | -      | 0.35    | -         | 3.34          | -         | [mmpose][pose_t_c]                      | [model][pose_t_m]                                           | 279           |

| Model                         | AP    | Gap    | Flop(G) | Remain(%) | Parameters(M) | Remain(%) | Config                                  | Download                                                    | Onnx_cpu(FPS) |
| ----------------------------- | ----- | ------ | ------- | --------- | ------------- | --------- | --------------------------------------- | ----------------------------------------------------------- | ------------- |
| rtmpose-s-aic-coco            | 0.722 | -      | 0.68    | -         | 5.47          | -         | [mmpose][pose_s_c]                      | [model][pose_s_m]                                           | 196           |
| rtmpose-s-aic-coco_pruned_act | 0.694 | -0.028 | 0.35    | 51.5%     | 3.43          | 62.7%     | [prune][rp_a_pc] \| [finetune][rp_a_fc] | [pruned][rp_sa_p] \| [finetuned][rp_sa_f] \| [log][rp_sa_l] | 272           |
| rtmpose-t-aic-coco            | 0.685 | -      | 0.35    | -         | 3.34          | -         | [mmpose][pose_t_c]                      | [model][pose_t_m]                                           | 279           |

- All FPS is test on the same machine with 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz.

## Get Started

We have three steps to apply GroupFisher to your model, including Prune, Finetune, Deploy.

Note: please use torch>=1.12, as we need fxtracer to parse the models automatically.

### Prune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
  {config_folder}/group_fisher_{normalization_type}_prune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

In the pruning config file. You have to fill some args as below.

```python
"""
_base_ (str): The path to your pretrained model checkpoint.
pretrained_path (str): The path to your pretrained model checkpoint.

interval (int): Interval between pruning two channels. You should ensure you
    can reach your target pruning ratio when the training ends.
normalization_type (str): GroupFisher uses two methods to normlized the channel
    importance, including ['flops','act']. The former uses flops, while the
    latter uses the memory occupation of activation feature maps.
lr_ratio (float): Ratio to decrease lr rate. As pruning progress is unstable,
    you need to decrease the original lr rate until the pruning training work
    steadly without getting nan.

target_flop_ratio (float): The target flop ratio to prune your model.
input_shape (Tuple): input shape to measure the flops.
"""
```

After the pruning process, you will get a checkpoint of the pruned model named flops\_{target_flop_ratio}.pth in your workdir.

### Finetune

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py 8 \
  --work-dir $WORK_DIR
```

There are also some args for you to fill in the config file as below.

```python
"""
_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""
```

After finetuning, except a checkpoint of the best model, there is also a fix_subnet.json, which records the pruned model structure. It will be used when deploying.

### Test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_test.sh \
   {config_folder}/group_fisher_{normalization_type}_finetune_{model_name}.py {checkpoint_path} 8
```

### Deploy

First, we assume you are fimilar to mmdeploy. For a pruned model, you only need to use the pruning deploy config to instead the pretrain config to deploy the pruned version of your model.

```bash
python {mmdeploy}/tools/deploy.py \
    {mmdeploy}/{mmdeploy_config}.py \
    {config_folder}/group_fisher_{normalization_type}_deploy_{model_name}.py \
    {path_to_finetuned_checkpoint}.pth \
    {mmdeploy}/tests/data/tiger.jpeg
```

The deploy config has some args as below:

```python
"""
_base_ (str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""
```

The divisor is important for the actual inference speed, and we suggest you to test it in \[1,2,4,8,16,32\] to find the fastest divisor.

## Implementation

All the modules of GroupFisher is placesded in mmrazor/implementations/pruning/group_fisher/.

| File                 | Module                                                               | Feature                                                                                 |
| -------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| algorithm.py         | GroupFisherAlgorithm                                                 | Dicide when to prune a channel according to the interval and the current iteration.     |
| mutator.py           | GroupFisherChannelMutator                                            | Select the unit with the channel of the minimal importance and to prune it.             |
| unit.py              | GroupFisherChannelUnit                                               | Compute fisher info                                                                     |
| ops.py <br> counters | GroupFisherConv2d <br> GroupFisherLinear <br> corresbonding counters | Collect model info to compute fisher info, including activation, grad and tensor shape. |

There are also some modules to support GroupFisher. These modules may be refactored and moved to other folders as common modules for all pruning algorithms.

| File                      | Module                                   | Feature                                                             |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| hook.py                   | PruningStructureHook<br>ResourceInfoHook | Display pruning Structure iteratively.                              |
| prune_sub_model.py        | GroupFisherSubModel                      | Convert a pruning algorithm(architecture) to a pruned static model. |
| prune_deploy_sub_model.py | GroupFisherDeploySubModel                | Init a pruned static model for mmdeploy.                            |

## Citation

```latex
@InProceedings{Liu:2021,
   TITLE      = {Group Fisher Pruning for Practical Network Compression},
   AUTHOR     = {Liu, Liyang
               AND Zhang, Shilong
               AND Kuang, Zhanghui
               AND Zhou, Aojun
               AND Xue, Jing-hao
               AND Wang, Xinjiang
               AND Chen, Yimin
               AND Yang, Wenming
               AND Liao, Qingmin
               AND Zhang, Wayne},
   BOOKTITLE  = {Proceedings of the 38th International Conference on Machine Learning},
   YEAR       = {2021},
   SERIES     = {Proceedings of Machine Learning Research},
   MONTH      = {18--24 Jul},
   PUBLISHER  = {PMLR},
}
```

<!-- model links
{model}_{prune_mode}_{file type}
model: r: resnet50, m: mobilenetv2, rt:retinanet
prune_mode: a: act, f: flops
file_type: p: pruned model, f:finetuned_model, l: log, pc: prune config, fc: finetune config.

repo link
{repo}_{model}_{file type}
 -->

[cls_m_c]: https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py
[cls_m_m]: https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth
[cls_r50_c]: https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet50_8xb32_in1k.py
[cls_r50_m]: https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth
[det_rt_c]: https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/retinanet/retinanet_r50_fpn_1x_coco.py
[det_rt_m]: https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
[m_a_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/act/group_fisher_act_finetune_mobilenet-v2_8xb32_in1k.pth
[m_a_fc]: ../../mmcls/group_fisher/mobilenet/group_fisher_act_finetune_mobilenet-v2_8xb32_in1k.py
[m_a_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/mobilenet/act/20230130_203443.json
[m_a_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/act/group_fisher_act_prune_mobilenet-v2_8xb32_in1k.pth
[m_a_pc]: ../../mmcls/group_fisher/mobilenet/group_fisher_act_prune_mobilenet-v2_8xb32_in1k.py
[m_f_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/flop/group_fisher_flops_finetune_mobilenet-v2_8xb32_in1k.pth
[m_f_fc]: ../../mmcls/group_fisher/mobilenet/group_fisher_flops_finetune_mobilenet-v2_8xb32_in1k.py
[m_f_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/mobilenet/flop/20230201_211550.json
[m_f_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/flop/group_fisher_flops_prune_mobilenet-v2_8xb32_in1k.pth
[m_f_pc]: ../../mmcls/group_fisher/mobilenet/group_fisher_flops_prune_mobilenet-v2_8xb32_in1k.py
[pose_s_c]: https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth
[pose_s_m]: https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth
[pose_t_c]: https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth
[pose_t_m]: https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth
[rp_a_fc]: ../../mmpose/group_fisher/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py
[rp_a_pc]: ../../mmpose/group_fisher/group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.py
[rp_sa_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sa_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.json
[rp_sa_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth
[rp_sc_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.pth
[rp_sc_l]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.json
[rp_sc_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.pth
[rt_a_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/act/group_fisher_act_finetune_retinanet_r50_fpn_1x_coco.pth
[rt_a_fc]: ../../mmdet/group_fisher/retinanet/group_fisher_act_finetune_retinanet_r50_fpn_1x_coco.py
[rt_a_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/retinanet/act/20230113_231904.json
[rt_a_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/act/group_fisher_act_prune_retinanet_r50_fpn_1x_coco.pth
[rt_a_pc]: ../../mmdet/group_fisher/retinanet/group_fisher_act_prune_retinanet_r50_fpn_1x_coco.py
[rt_f_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/flops/group_fisher_flops_finetune_retinanet_r50_fpn_1x_coco.pth
[rt_f_fc]: ../../mmdet/group_fisher/retinanet/group_fisher_flops_finetune_retinanet_r50_fpn_1x_coco.py
[rt_f_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/retinanet/flops/20230129_101502.json
[rt_f_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/flops/group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.pth
[rt_f_pc]: ../../mmdet/group_fisher/retinanet/group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.py
[r_a_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/act/group_fisher_act_finetune_resnet50_8xb32_in1k.pth
[r_a_fc]: ../../mmcls/group_fisher/resnet50/group_fisher_act_finetune_resnet50_8xb32_in1k.py
[r_a_fc_kd]: ../../mmcls/group_fisher/resnet50/group_fisher_act_finetune_resnet50_8xb32_in1k_dist.py
[r_a_f_kd]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/group_fisher_act_finetune_resnet50_8xb32_in1k_dist.pth
[r_a_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/resnet50/act/20230130_175426.json
[r_a_l_kd]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/group_fisher_act_finetune_resnet50_8xb32_in1k_dist.json
[r_a_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/act/group_fisher_act_prune_resnet50_8xb32_in1k.pth
[r_a_pc]: ../../mmcls/group_fisher/resnet50/group_fisher_act_prune_resnet50_8xb32_in1k.py
[r_f_f]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/flops/group_fisher_flops_finetune_resnet50_8xb32_in1k.pth
[r_f_fc]: ../../mmcls/group_fisher/resnet50/group_fisher_flops_finetune_resnet50_8xb32_in1k.py
[r_f_l]: https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/group_fisher/resnet50/flops/20230129_190931.json
[r_f_p]: https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/flops/group_fisher_flops_prune_resnet50_8xb32_in1k.pth
[r_f_pc]: ../../mmcls/group_fisher/resnet50/group_fisher_flops_prune_resnet50_8xb32_in1k.py
