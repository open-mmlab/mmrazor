# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.005))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[50, 100], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=150)
# val_cfg = dict(interval=1)  # validate every epoch
val_cfg = dict()  # validate every epoch
test_cfg = dict()
