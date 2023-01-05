set -e

save_dir=work_dirs/Zen_NAS_cifar_params1M
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

./command.sh search_kk 1 1 "python mmrazor/models/architectures/backbones/masternet.py --gpu 0 \
  --zero_shot_score Zen \
  --search_space mmrazor/models/architectures/backbones/SearchSpace/search_space_XXBL.py \
  --budget_model_size 1e6 \
  --max_layers 18 \
  --batch_size 64 \
  --input_image_size 32 \
  --plainnet_struct_txt work_dirs/Zen_NAS_cifar_params1M/init_plainnet.txt \
  --num_classes 100 \
  --evolution_max_iter 480000 \
  --population_size 512 \
  --save_dir work_dirs/Zen_NAS_cifar_params1M"


# 1ms
# ./command.sh search_kk 1 1 "python mmrazor/models/architectures/backbones/masternet.py --gpu 0 \
#   --zero_shot_score Zen \
#   --search_space mmrazor/models/architectures/backbones/SearchSpace/search_space_XXBL.py \
#   --budget_latency 1e-4 \
#   --max_layers 10 \
#   --batch_size 64 \
#   --input_image_size 224 \
#   --plainnet_struct_txt ./work_dirs/1ms/init_plainnet.txt \
#   --num_classes 1000 \
#   --evolution_max_iter 20000 \
#   --population_size 512 \
#   --save_dir ./work_dirs/1ms"