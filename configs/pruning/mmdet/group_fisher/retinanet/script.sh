# act mode
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_act_prune_retinanet_r50_fpn_1x_coco.py 8
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_act_finetune_retinanet_r50_fpn_1x_coco.py 8

# flops mode
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.py 8
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_flops_finetune_retinanet_r50_fpn_1x_coco.py 8



# deploy act mode

razor_config=configs/pruning/mmdet/group_fisher/retinanet/group_fisher_act_deploy_retinanet_r50_fpn_1x_coco.py
deploy_config=mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/act/group_fisher_act_finetune_retinanet_r50_fpn_1x_coco.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 800x1248 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100

# deploy flop mode

razor_config=configs/pruning/mmdet/group_fisher/retinanet/group_fisher_flops_deploy_retinanet_r50_fpn_1x_coco.py
deploy_config=mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/flops/group_fisher_flops_finetune_retinanet_r50_fpn_1x_coco.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 800x1248 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100
