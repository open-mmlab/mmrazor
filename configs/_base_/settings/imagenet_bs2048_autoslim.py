_base_ = [
    './imagenet_bs1024_spos.py',
]

train_dataloader = dict(batch_size=256)

val_dataloader = dict(batch_size=256)

test_dataloader = dict(batch_size=256)
