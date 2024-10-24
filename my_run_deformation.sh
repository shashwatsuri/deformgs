CUDA_VISIBLE_DEVICES=0 python train.py \
-s "/scratch-ssd/Datasets/deformation/nerf/" \
--port 6059 \
--expname "hemisphere_ts10_bigkplanes" \
--configs "arguments/mdnerf-dataset/hemisphere.py" \
--lambda_w 2000 \
--lambda_rigidity 0.0 \
--lambda_spring 0.0 \
--lambda_momentum 0.03 \
--lambda_velocity 0.0 \
--view_skip 1 \
--time_skip 10 \
--k_nearest 20 \
--lambda_isometric 0.3 \
--reg_iter 11000 \
--staticfying_from 10000 \
--use_wandb \
--wandb_project "deforming_hemisphere" \
--wandb_name "deforming_hemisphere" \
--white_background
