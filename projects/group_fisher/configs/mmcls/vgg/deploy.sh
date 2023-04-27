python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_static.py \
    projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_deploy.py \
    ./work_dirs/vgg_group_fisher_finetune/best_accuracy/top1_epoch_142.pth \
    ./mmdeploy/demo/resources/face.png  \
    --work-dir work_dirs/mmdeploy_model/ \
    --device cpu \
    --dump-info

python mmdeploy/tools/profiler.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_static.py \
    projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_deploy.py \
    mmdeploy/resources/ \
    --model ./work_dirs/mmdeploy_model/end2end.onnx \
    --shape 32x32 \
    --device cpu \
    --warmup 50 \
    --num-iter 200

python mmdeploy/tools/test.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_static.py \
    projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_deploy.py \
    --model ./work_dirs/mmdeploy_model/end2end.onnx \
    --device cpu \
