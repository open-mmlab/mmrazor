
# export pruned checkpoint example

python ./tools/pruning/get_static_model_from_algorithm.py configs/pruning/mmcls/l1-norm/l1-norm_resnet34_8xb32_in1k_a.py https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_a.pth -o ./work_dirs/norm_resnet34_8xb32_in1k_a

# deploy example

razor_config=configs/pruning/mmcls/l1-norm/l1-norm_resnet34_8xb32_in1k_a_deploy.py
deploy_config=mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py
static_model_checkpoint_path=path/to/pruend/checkpoint

python mmdeploy/tools/deploy.py $deploy_config \
    $razor_config \
    $static_model_checkpoint_path \
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
