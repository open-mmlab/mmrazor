_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']

data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}
architecture = _base_.model
architecture.update({
    'init_cfg': {
        'type':
        'Pretrained',
        'checkpoint':
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
    }
})

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ChexAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ChexMutator',
        channel_unit_cfg=dict(
            type='ChexUnit', default_args=dict(choice_mode='number', )),
        channel_ratio=0.7,
    ),
    delta_t=2,
    total_steps=60,
    init_growth_rate=0.3,
)
custom_hooks = [{'type': 'mmrazor.ChexHook'}]
