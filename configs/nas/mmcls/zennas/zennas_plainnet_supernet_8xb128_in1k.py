_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    'mmcls::_base_/default_runtime.py',
]

# optim_wrapper=None
model_wrapper_cfg = None
optim_wrapper = dict(_delete_=True, type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01))

nas_backbone = dict(
    _scope_='mmrazor',
    type='MasterNet',
    fix_subnet='./work_dirs/1ms/init_plainnet.txt',
    no_create=True,
    num_classes=1000)

# model
supernet = dict(
    type='ImageClassifier',
    data_preprocessor=_base_.preprocess_cfg,
    backbone=nas_backbone)

model = dict(
    type='mmrazor.ZenNAS',
    architecture=supernet
)

# find_unused_parameters = True
