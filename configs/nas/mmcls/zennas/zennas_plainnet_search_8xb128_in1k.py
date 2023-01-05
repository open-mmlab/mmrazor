_base_ = ['./zennas_plainnet_supernet_8xb128_in1k.py']

# model = dict(norm_training=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.ZeroShotLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    search_space='mmrazor/models/architectures/backbones/SearchSpace/search_space_XXBL.py',
    plainnet_struct_txt='./work_dirs/1ms/init_plainnet.txt',
    max_epochs=480000, # evolution_max_iter
    population_size=512,
    num_classes=1000,
)