model = dict(
    _scope_='mmrazor',
    type='mmrazor.ItePruneAlgorithm',
    architecture=dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='VGGPruning', num_classes=10),
        # neck=dict(), # do not use neck
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        ),
    ),
    target_pruning_ratio={
        'features_conv4_(0, 128)_out_2_in_1': 0.8,
        'features_conv15_(0, 512)_out_2_in_1': 0.8,
        'features_conv8_(0, 256)_out_2_in_1': 0.8,
        'features_conv0_(0, 64)_out_2_in_1': 0.8,
        'features_conv12_(0, 512)_out_2_in_1': 0.8,
        'features_conv1_(0, 64)_out_2_in_1': 0.8,
        'features_conv6_(0, 256)_out_2_in_1': 0.8,
        'features_conv16_(0, 512)_out_2_in_1': 0.8,
        'features_conv10_(0, 512)_out_2_in_1': 0.8,
        'features_conv3_(0, 128)_out_2_in_1': 0.8,
        'features_conv14_(0, 512)_out_2_in_1': 0.8,
        'features_conv7_(0, 256)_out_2_in_1': 0.8,
        'features_conv11_(0, 512)_out_2_in_1': 0.8
    })
