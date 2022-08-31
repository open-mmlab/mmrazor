_base_ = [
    './factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_pretrain.py'
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)

model = dict(
    calculate_student_loss=True,
    student_trainable=True,
    distiller=dict(
        distill_losses=dict(loss_s4=dict(type='FTLoss', loss_weight=1.0)),
        connectors=dict(loss_s4_tfeat=dict(phase='train')),
        loss_forward_mappings=dict(
            _delete_=True,
            loss_s4=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='loss_s4_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_s4',
                    connector='loss_s4_tfeat'),
            ))),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'work_dirs/factor-transfer_backbone_resnet50_resnet18_8xb16_cifar10_pretrain/epoch_20.pth'  # noqa: E501
    ))
