_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

# model = dict(fix_subnet='configs/nas/mmcls/bignas/subnet.yaml')
model = dict(
    fix_subnet=
    '/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/work_dirs/1206_search_a6/best_fix_subnet.yaml'
)

# load_from = '/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/.vscode/mmrazor_bignas_pr/save_ckpt/final_subnet_20221206_1639.pth'

test_cfg = dict(_delete_=True)
