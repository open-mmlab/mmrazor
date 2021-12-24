# Train a model with our algorithms

Currently our algorithms support [mmclassification](https://mmclassification.readthedocs.io/en/latest/), [mmdetection ](https://mmdetection.readthedocs.io/en/latest/)and [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/). **Before running our algorithms, you may need to prepare the datasets according to the instructions in the  corresponding  document.**

**Note**:

- Since our algorithms **have the same interface for all three tasks**, in the following introduction, we use `${task}` to represent one of `mmcls`、`mmdet` and `mmseg`.
- We dynamically pass arguments `cfg-options` (e.g., `mutable_cfg` in nas algorithm or `channel_cfg` in pruning algorithm)  to **avoid the need for a config for each subnet or checkpoint**. If you want to specify different subnets for retraining or testing, you just need to change this arguments.

## NAS

Three are three steps to start neural network search(NAS), including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**.

### Supernet Pre-training

```Bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} [optional arguments]
```

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

### Search for Subnet on The Trained Supernet

```Bash
python tools/${task}/search_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```

### Subnet Retraining

```bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.mutable_cfg=${MUTABLE_CFG_PATH} [optional arguments]
```

- `MUTABLE_CFG_PATH`: Path of `mutable_cfg`. `mutable_cfg` represents **config for mutable of the subnet searched out**, used to specify different subnets for retraining. An example for `mutable_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/README.md#subnet-retraining-on-imagenet).

## Pruning

Pruning has four steps, including **supernet pre-training**, **search for subnet on the trained supernet**, **subnet retraining** and **split checkpoint**. The commands of the first two steps are similar to NAS, except that we need to use `CONFIG_FILE` of Pruning here. The commands of the two other steps are as follows.

### Subnet Retraining

```bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.channel_cfg=${CHANNEL_CFG_PATH} [optional arguments]
```

Different from NAS, the argument that needs to be specified here is `channel_cfg` instead of `mutable_cfg`.

- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/README.md#subnet-retraining-on-imagenet).

## Distillation

There is only one step to start knowledge distillation.

```Bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.distiller.teacher.init_cfg.type=Pretrained algorithm.distiller.teacher.init_cfg.checkpoint=${TEACHER_CHECKPOINT_PATH} [optional arguments]
```

- `TEACHER_CHECKPOINT_PATH`: Path of `teacher_checkpoint`. `teacher_checkpoint` represents **checkpoint of teacher model**, used to specify different checkpoints for distillation.
