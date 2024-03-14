#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export isometry=0.316227766

port=6027 

python3 render_experimental.py --model_path "output/mimic/mimic_1_no_reg" --skip_train \
        --configs arguments/mdnerf-dataset/cube.py --view_skip 10 --time_skip 1 --log_deform 

 