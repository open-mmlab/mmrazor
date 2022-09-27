_base_ = ['mmcls::deit/deit-base_pt-16xb64_in1k.py']

# student settings
student = _base_.model
student.backbone.type = 'DistilledVisionTransformer'
student.head.type = 'DeiTClsHead'
student.head.loss.loss_weight = 0.5

data_preprocessor = dict(type='mmcls.ClsDataPreprocessor')

# teacher settings
teacher = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='regnety_160',
        # checkpoint_path='/mnt/lustre/caoweihan.p/ckpt/regnety_160-a5fe301d.pth'
        checkpoint_path=
        r'G:\projects\research\checkpoint\regnety_160-a5fe301d.pth',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=3024,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
            loss_weight=0.5),
        topk=(1, 5),
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='/mnt/lustre/caoweihan.p/ckpt/regnety_160-a5fe301d.pth',
            checkpoint=
            r'G:\projects\research\checkpoint\regnety_160-a5fe301d.pth',
            prefix='head.'),
    ))

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='SingleTeacherDistill',
    architecture=student,
    teacher=teacher,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.layers.head_dist')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_distill=dict(
                type='KLDivergence',
                tau=1.,
                loss_weight=0.5,
                reduction='batchmean',
                teacher_detach=True)),
        loss_forward_mappings=dict(
            loss_distill=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=3,
        out_dir='s3://caoweihan/deit2.0',
        save_best='acc',
        rule='greater'))
