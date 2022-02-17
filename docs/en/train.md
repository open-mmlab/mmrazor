# Train a model with our algorithms

Currently our algorithms support [mmclassification](https://mmclassification.readthedocs.io/en/latest/), [mmdetection ](https://mmdetection.readthedocs.io/en/latest/)and [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/). **Before running our algorithms, you may need to prepare the datasets according to the instructions in the  corresponding  document.**

**Note**:

- Since our algorithms **have the same interface for all three tasks**, in the following introduction, we use `${task}` to represent one of `mmcls`、`mmdet` and `mmseg`.
- We dynamically pass arguments `cfg-options` (e.g., `mutable_cfg` in nas algorithm or `channel_cfg` in pruning algorithm)  to **avoid the need for a config for each subnet or checkpoint**. If you want to specify different subnets for retraining or testing, you just need to change this arguments.

## NAS

There are three steps to start neural network search(NAS), including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**.

### Supernet Pre-training

```Bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} [optional arguments]
```

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

For example,

<pre>
python ./tools/mmcls/train_mmcls.py \
  configs/nas/spos/spos_supernet_shufflenetv2_8xb128_in1k.py \
  --work-dir $WORK_DIR
</pre>

### Search for Subnet on The Trained Supernet

```Bash
python tools/${task}/search_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```

For example,

<pre>
python ./tools/mmcls/search_mmcls.py \
  configs/nas/spos/spos_evolution_search_shufflenetv2_8xb2048_in1k.py \
  $STEP1_CKPT \
  --work-dir $WORK_DIR
</pre>

### Subnet Retraining

```bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.mutable_cfg=${MUTABLE_CFG_PATH} [optional arguments]
```

- `MUTABLE_CFG_PATH`: Path of `mutable_cfg`. `mutable_cfg` represents **config for mutable of the subnet searched out**, used to specify different subnets for retraining. An example for `mutable_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/README.md#subnet-retraining-on-imagenet).

For example,

<pre>
python ./tools/mmcls/train_mmcls.py \
  configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py \
  --work-dir $WORK_DIR \
  --cfg-options algorithm.mutable_cfg=configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml
</pre>

## Pruning

Pruning has three steps, including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**. The commands of the first two steps are similar to NAS, except that we need to use `CONFIG_FILE` of Pruning here. The commands of the **subnet retraining** is as follows.

### Subnet Retraining

```bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.channel_cfg=${CHANNEL_CFG_PATH} [optional arguments]
```

Different from NAS, the argument that needs to be specified here is `channel_cfg` instead of `mutable_cfg`.

- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/README.md#subnet-retraining-on-imagenet).

For example,

<pre>
python ./tools/mmcls/train_mmcls.py \
  configs/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  --work-dir <em>your_work_dir</em> \
  --cfg-options algorithm.channel_cfg=configs/pruning/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml,configs/pruning/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml,configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml
</pre>

## Distillation

There is only one step to start knowledge distillation.

```Bash
python tools/${task}/train_${task}.py ${CONFIG_FILE} --cfg-options algorithm.distiller.teacher.init_cfg.type=Pretrained algorithm.distiller.teacher.init_cfg.checkpoint=${TEACHER_CHECKPOINT_PATH} [optional arguments]
```

- `TEACHER_CHECKPOINT_PATH`: Path of `teacher_checkpoint`. `teacher_checkpoint` represents **checkpoint of teacher model**, used to specify different checkpoints for distillation.

For example,

<pre>
python ./tools/mmdet/train_mmdet.py \
  configs/distill/cwd/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k.py \
  --work-dir <em>your_work_dir</em>
</pre>
