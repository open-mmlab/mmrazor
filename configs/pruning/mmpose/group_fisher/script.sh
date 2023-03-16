# deploy rtmpose-s_pruned_act

razor_config=configs/pruning/mmpose/group_fisher/group_fisher_deploy_rtmpose-s_8xb256-420e_coco-256x192.py
deploy_config=mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100

# deploy rtmpose-s-aic-coco_pruned_act

razor_config=configs/pruning/mmpose/group_fisher/group_fisher_deploy_rtmpose-s_8xb256-420e_aic-coco-256x192.py
deploy_config=mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth \
    mmdeploy/tests/data/tiger.jpeg \
    --work-dir ./work_dirs/mmdeploy

python mmdeploy/tools/profiler.py $deploy_config \
    $razor_config \
    mmdeploy/demo/resources \
    --model ./work_dirs/mmdeploy/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --num-iter 1000 \
    --warmup 100
