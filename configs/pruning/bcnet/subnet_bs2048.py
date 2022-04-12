_base_ = [
    './supernet.py',
]

algorithm = dict(
    distiller=None,
    retraining=True,
    bn_training_mode=False,
)

custom_hooks = None

data = dict(samples_per_gpu=256)
optimizer = dict(lr=0.8)
