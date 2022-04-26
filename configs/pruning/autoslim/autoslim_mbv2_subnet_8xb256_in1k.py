_base_ = [
    './autoslim_mbv2_supernet_8xb256_in1k.py',
]

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0)))

# FIXME: you may replace this with the channel_cfg searched by yourself
channel_cfg = [
    'configs/pruning/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml',
    'configs/pruning/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml',
    'configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml'
]

algorithm = dict(
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    retraining=True,
    bn_training_mode=False,
    channel_cfg=channel_cfg)

runner = dict(type='EpochBasedRunner', max_epochs=300)

find_unused_parameters = True
