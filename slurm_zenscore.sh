set -e

./command.sh search_kk 1 1 "python mmrazor/models/architectures/backbones/masternet.py --gpu 0 \
  --zero_shot_score Zen \
  --search_space mmrazor/models/architectures/backbones/SearchSpace/search_space_XXBL.py \
  --budget_latency 1e-4 \
  --max_layers 10 \
  --batch_size 64 \
  --input_image_size 224 \
  --plainnet_struct_txt ./work_dirs/1ms/init_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter 20000 \
  --population_size 512 \
  --save_dir ./work_dirs/1ms"