_base_ = [
    './vgg_model.py', './cifar10_bs16.py', './cifar10_bs128.py',
    './default_runtime.py'
]
custom_imports = dict(imports=['projects'])
