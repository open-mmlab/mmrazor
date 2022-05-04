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
    'https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.53M_acc-74.23_20211222-e5208bbd_channel_cfg.yaml',  # noqa: E501
    'https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.32M_acc-72.73_20211222-b5b0b33c_channel_cfg.yaml',  # noqa: E501
    'https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.22M_acc-71.39_20211222-43117c7b_channel_cfg.yaml'  # noqa: E501
]

algorithm = dict(
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    retraining=True,
    bn_training_mode=False,
    channel_cfg=channel_cfg)

runner = dict(type='EpochBasedRunner', max_epochs=300)

find_unused_parameters = True
