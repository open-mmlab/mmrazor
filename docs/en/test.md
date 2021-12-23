# Test a model

## NAS

To test nas method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.mutable_cfg=${MUTABLE_CFG_PATH} [optional arguments]
```

- `MUTABLE_CFG_PATH`: Path of `mutable_cfg`. `mutable_cfg` represents **config for mutable of the subnet searched out**, used to specify different subnets for testing. An example for `mutable_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml).

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

## Pruning

### Split Checkpoint(Optional)
If you train a slimmable model during retrain, checkpoints of different subnets are
actually fused in only one checkpoint. You can split this checkpoint to
multiple independent checkpoints by using following command

```bash
python tools/model_converters/split_checkpoint.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --channel-cfgs ${CHANNEL_CFG_PATH} [optional arguments]
```

- `CHANNEL_CFG_PATH`: A list of paths of `channel_cfg`. For example, when you
retrain a slimmable model, your command will be like `--cfg-options algorithm.channel_cfg=cfg1,cfg2,cfg3`.
And your command here should be `--channel-cfgs cfg1 cfg2 cfg3`. The order of them should be the same.

### Test

To test pruning method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.channel_cfg=${CHANNEL_CFG_PATH} [optional arguments]
```

- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/README.md#test-a-subnet).

## Distillation

To test distillation method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```
