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

algorithm = dict(
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    retraining=True,
    bn_training_mode=False,
)

runner = dict(type='EpochBasedRunner', max_epochs=300)

find_unused_parameters = True
