# act mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_act_prune_mobilenet-v2_8xb32_in1k.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_act_finetune_mobilenet-v2_8xb32_in1k.py 8

# flops mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_flops_prune_mobilenet-v2_8xb32_in1k.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_flops_finetune_mobilenet-v2_8xb32_in1k.py 8


# deploy act mode
razor_config=configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_act_deploy_mobilenet-v2_8xb32_in1k.py
deploy_config=mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/act/group_fisher_act_finetune_mobilenet-v2_8xb32_in1k.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 224x224 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100

# deploy flop mode

razor_config=configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_flops_deploy_mobilenet-v2_8xb32_in1k.py
deploy_config=mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/mobilenet/flop/group_fisher_flops_finetune_mobilenet-v2_8xb32_in1k.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 224x224 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100
