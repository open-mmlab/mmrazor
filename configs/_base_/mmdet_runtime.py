checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Default setting for scaling LR automatically
#   - The flag `auto_scale_lr` means enable scaling LR automatically
#       or not by default.
#   - `default_batch_size` = (8 GPUs) x (2 samples per GPU).
#   - `default_initial_lr` = The LR by default.
auto_scale_lr_config = dict(
    auto_scale_lr=False, default_batch_size=16, default_initial_lr=0.01)
