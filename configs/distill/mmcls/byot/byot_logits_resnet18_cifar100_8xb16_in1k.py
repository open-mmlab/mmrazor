_base_ = [
    '../../../_base_/datasets/mmcls/cifar100_bs16_auto_aug.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[80, 160, 240], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)

model = dict(
    _scope_='mmrazor',
    type='SelfDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[129.304, 124.070, 112.434],
        std=[68.170, 65.392, 70.418],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=dict(
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='mmcls.ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0))),
    distiller=dict(
        type='BYOTDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.1.relu'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc'),
            neck_gap=dict(type='ModuleOutputs', source='neck.gap'),
            gt_labels=dict(type='ModuleInputs', source='head.loss_module')),
        distill_losses=dict(
            loss_fet_1=dict(
                type='L2Loss', normalize=False, loss_weight=0.03, dist=True),
            loss_label_1=dict(type='mmcls.CrossEntropyLoss', loss_weight=0.7),
            loss_softl_1=dict(type='KLDivergence', tau=3, loss_weight=0.3),
            loss_fet_2=dict(
                type='L2Loss', normalize=False, loss_weight=0.03, dist=True),
            loss_label_2=dict(type='mmcls.CrossEntropyLoss', loss_weight=0.7),
            loss_softl_2=dict(type='KLDivergence', tau=3, loss_weight=0.3),
            loss_fet_3=dict(
                type='L2Loss', normalize=False, loss_weight=0., dist=True),
            loss_label_3=dict(type='mmcls.CrossEntropyLoss', loss_weight=0.7),
            loss_softl_3=dict(type='KLDivergence', tau=3, loss_weight=0.3)),
        connectors=dict(
            loss_s1_sfeat=dict(
                type='BYOTConnector',
                in_channel=64,
                out_channel=512,
                expansion=1,
                kernel_size=3,
                stride=2,
                num_classes=100),
            loss_s2_sfeat=dict(
                type='BYOTConnector',
                in_channel=128,
                out_channel=512,
                expansion=1,
                kernel_size=3,
                stride=2,
                num_classes=100),
            loss_s3_sfeat=dict(
                type='BYOTConnector',
                in_channel=256,
                out_channel=512,
                expansion=1,
                kernel_size=3,
                stride=2,
                num_classes=100)),
        loss_forward_mappings=dict(
            loss_fet_1=dict(
                s_feature=dict(
                    recorder='bb_s1',
                    from_student=True,
                    connector='loss_s1_sfeat',
                    connector_idx=0),
                t_feature=dict(recorder='neck_gap', from_student=False)),
            loss_label_1=dict(
                cls_score=dict(
                    recorder='bb_s1',
                    from_student=True,
                    connector='loss_s1_sfeat',
                    connector_idx=1),
                label=dict(
                    recorder='gt_labels', from_student=False, data_idx=1)),
            loss_softl_1=dict(
                preds_S=dict(
                    recorder='bb_s1',
                    from_student=True,
                    connector='loss_s1_sfeat',
                    connector_idx=1),
                preds_T=dict(recorder='fc', from_student=False)),
            loss_fet_2=dict(
                s_feature=dict(
                    recorder='bb_s2',
                    from_student=True,
                    connector='loss_s2_sfeat',
                    connector_idx=0),
                t_feature=dict(recorder='neck_gap', from_student=False)),
            loss_label_2=dict(
                cls_score=dict(
                    recorder='bb_s2',
                    from_student=True,
                    connector='loss_s2_sfeat',
                    connector_idx=1),
                label=dict(
                    recorder='gt_labels', from_student=False, data_idx=1)),
            loss_softl_2=dict(
                preds_S=dict(
                    recorder='bb_s2',
                    from_student=True,
                    connector='loss_s2_sfeat',
                    connector_idx=1),
                preds_T=dict(recorder='fc', from_student=False)),
            loss_fet_3=dict(
                s_feature=dict(
                    recorder='bb_s3',
                    from_student=True,
                    connector='loss_s3_sfeat',
                    connector_idx=0),
                t_feature=dict(recorder='neck_gap', from_student=False)),
            loss_label_3=dict(
                cls_score=dict(
                    recorder='bb_s3',
                    from_student=True,
                    connector='loss_s3_sfeat',
                    connector_idx=1),
                label=dict(
                    recorder='gt_labels', from_student=False, data_idx=1)),
            loss_softl_3=dict(
                preds_S=dict(
                    recorder='bb_s3',
                    from_student=True,
                    connector='loss_s3_sfeat',
                    connector_idx=1),
                preds_T=dict(recorder='fc', from_student=False)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SelfDistillValLoop')
