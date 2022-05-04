# Train different type algorithms

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

We note that instead of using ``--cfg-options``, you can also directly modify ``configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py`` like this:

<pre>
mutable_cfg = 'configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml'
algorithm = dict(..., mutable_cfg=mutable_cfg)
</pre>

## Pruning

Pruning has three steps, including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**. The commands of the first two steps are similar to NAS, except that we need to use `CONFIG_FILE` of Pruning here. The commands of the **subnet retraining** are as follows.

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
python ./tools/mmseg/train_mmseg.py \
  configs/distill/cwd/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k.py \
  --work-dir <em>your_work_dir</em> \
  --cfg-options algorithm.distiller.teacher.init_cfg.type=Pretrained algorithm.distiller.teacher.init_cfg.checkpoint=https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth
</pre>

# Train with different devices

**Note**: The default learning rate in config files is for 8 GPUs. If using different number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `new_lr = old_lr * new_ngpus / old_ngpus`. We recommend to use `tools/xxx/dist_train.sh` even with 1 gpu, since some methods do not support non-distributed training.

### Training with CPU

```shell
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py ${CONFIG_FILE}
```

**Note**: We do not recommend users to use CPU for training because it is too slow and some algorithms are using `SyncBN` which is based on distributed training. We support this feature to allow users to debug on machines without GPU for convenience.

### Train with single/multiple GPUs

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work_dir ${YOUR_WORK_DIR} [optional arguments]
```


**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s ${YOUR_WORK_DIRS} ${MMRAZOR}/work_dirs
```

Alternatively, if you run MMRazor on a cluster managed with [slurm](https://slurm.schedmd.com/):

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} sh tools/xxx/slurm_train_xxx.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${YOUR_WORK_DIR} [optional arguments]
```

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/xxx/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/xxx/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

If you launch with slurm, the command is the same as that on single machine described above, but you need refer to [slurm_train.sh](https://github.com/open-mmlab/mmselfsup/blob/master/tools/slurm_train.sh) to set appropriate parameters and environment variables.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/xxx/dist_train.sh ${CONFIG_FILE} 4 --work_dir tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/xxx/dist_train.sh ${CONFIG_FILE} 4 --work_dir tmp_work_dir_2
```

If you use launch training jobs with slurm, you have two options to set different communication ports:

Option 1:

In `config1.py`:

```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`:

```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with config1.py and config2.py.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/xxx/slurm_train_xxx.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/xxx/slurm_train_xxx.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```

Option 2:

You can set different communication ports without the need to modify the configuration file, but have to set the `cfg-options` to overwrite the default port in configuration file.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/xxx/slurm_train_xxx.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/xxx/slurm_train_xxx.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
```
