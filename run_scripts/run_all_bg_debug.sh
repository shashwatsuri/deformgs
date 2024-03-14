#!/bin/bash
export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export ISOMETRY=0.01
export LAMBDA_VELOCITY=1.0

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_7"
export SCENE_5="scene_5"
export SCENE_6="scene_6"

port=6027 
for SCENE in $SCENE_1;
#for SCENE in $SCENE_1;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/final_scenes_bg/${SCENE}" --port $port --expname "final_scenes_bg_20m_range_exp_iso/${SCENE}_l1_velocity" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.0 --staticfying_from 1 \
        --k_nearest 5 --lambda_isometric $isometry --time_skip 1 --view_skip 50 --reg_iter 1 --lambda_velocity $LAMBDA_VELOCITY --coarse_t0 --no_coarse
        # add one to port
        port=$((port+1))
    done
done