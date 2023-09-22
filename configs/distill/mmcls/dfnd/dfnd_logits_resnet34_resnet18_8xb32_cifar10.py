_base_ = ['mmcls::_base_/default_runtime.py']

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[320, 640], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=32),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='/cache/data/imagenet/',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

test_pipeline = [
    dict(type='PackClsInputs'),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='/cache/data/cifar',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

teacher_ckpt = '/cache/models/resnet_model.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='DFNDDistill',
    calculate_student_loss=False,
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    val_data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teacher=dict(
        cfg_path='mmcls::resnet/resnet34_8xb16_cifar10.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(
                type='DFNDLoss',
                tau=4,
                loss_weight=1,
                num_classes=10,
                batch_select=0.5)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.DFNDValLoop')
