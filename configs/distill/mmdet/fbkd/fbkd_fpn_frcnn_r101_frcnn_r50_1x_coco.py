_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        pretrained=True),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py',
        pretrained=False),
    teacher_ckpt='faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth',
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv')),
        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s0_sfeat=dict(
                type='FBKDStudentConnector',
                in_channel=256,
                inter_channel=64,
                downsample_stride=8),
            loss_s0_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channel=256,
                inter_channel=64,
                downsample_stride=8),
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channel=256,
                inter_channel=64,
                downsample_stride=4),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channel=256,
                downsample_stride=4),
            loss_s2_sfeat=dict(type='FBKDStudentConnector', in_channel=256),
            loss_s2_tfeat=dict(type='FBKDTeacherConnector', in_channel=256),
            loss_s3_sfeat=dict(type='FBKDStudentConnector', in_channel=256),
            loss_s3_tfeat=dict(type='FBKDTeacherConnector', in_channel=256)),
        loss_forward_mappings=dict(
            loss_s0=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='neck_s0',
                    connector='loss_s0_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='neck_s0',
                    connector='loss_s0_tfeat')),
            loss_s1=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='neck_s1',
                    connector='loss_s1_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='neck_s1',
                    connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='neck_s2',
                    connector='loss_s2_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='neck_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='neck_s3',
                    connector='loss_s3_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='neck_s3',
                    connector='loss_s3_tfeat')))))

find_unused_parameters = True

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/': 's3://openmmlab/datasets/detection/',
        'data/': 's3://openmmlab/datasets/detection/'
    }))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
