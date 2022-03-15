_base_ = [
    './supernet.py',
]

algorithm = dict(
    distiller=None,
    retraining=True,
    bn_training_mode=False,
)

custom_hooks = None
