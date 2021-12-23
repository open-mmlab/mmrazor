# Test a model

## NAS

To test nas method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.mutable_cfg=${MUTABLE_CFG_PATH} [optional arguments]
```

- `MUTABLE_CFG_PATH`: Path of `mutable_cfg`. `mutable_cfg` represents **config for mutable of the subnet searched out**, used to specify different subnets for testing. An example for `mutable_cfg` can be found [here](/configs/nas/spos/SPOS_SHUFFLENET_300M.yaml).

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

## Pruning

To test pruning method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.channel_cfg=${CHANNEL_CFG_PATH} [optional arguments]
```

- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](/configs/pruning/autoslim/autoslim_220M.yaml), and the usage can be found [here](/configs/pruning/autoslim/README.md#test-a-subnet).

## Distillation

To test distillation method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```
