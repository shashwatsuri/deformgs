CUDA_VISIBLE_DEVICES=1 python render_experimental.py \
--iteration 30000 \
--model_path "/scratch-ssd/Repos/deformgs/output/hemisphere/c3t1_masks_flip" \
--configs "arguments/mdnerf-dataset/hemisphere.py" \
--skip_test \
--skip_video \
--time_skip 1 \
--view_ids 0 \
--scale 0.5 \
