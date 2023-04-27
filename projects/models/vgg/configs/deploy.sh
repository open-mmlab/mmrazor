
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_static.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
   ./work_dirs/pretrained/vgg_pretrained.pth \
    ./mmdeploy/demo/resources/face.png  \
    --work-dir work_dirs/mmdeploy_model/ \
    --device cpu \
    --dump-info

python mmdeploy/tools/profiler.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_static.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
    mmdeploy/resources/ \
    --model ./work_dirs/mmdeploy_model/end2end.onnx \
    --shape 32x32 \
    --device cpu \
    --warmup 50 \
    --num-iter 200
