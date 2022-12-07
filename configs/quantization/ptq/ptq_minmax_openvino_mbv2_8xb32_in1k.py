_base_ = ['mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py']

test_cfg = dict(type='mmrazor.PTQLoop', )

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    float_checkpoint=  # noqa: E251
    '/mnt/petrelfs/caoweihan.p/ckpt/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',  # noqa: E501
    quantizer=dict(
        type='mmrazor.OpenvinoQuantizer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.MinMaxObserver'),
            a_observer=dict(type='mmrazor.EMAMinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=True,
                is_pot_scale=False,
            ),
            a_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False),
        )))

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)
