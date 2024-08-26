export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=1.0
export LAMBDA_MOMENTUM=0.5

export FOLD_CLOTH="fold_cloth_short"
export CLOTH_FILTERED="cloth_filtered"
export XARM_FOLD="xarm_fold"

port=6067
for SCENE in $XARM_FOLD; 
do
    python3 train.py -s "data/panopto/${SCENE}" --port 6070 --expname "panopto/${SCENE}_updated_params" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
    --lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 3 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 4 --time_skip 1  --scale 0.25 \
    --staticfying_from 5000 --staticfying_interval 100 --staticfying_until 15000 --lambda_mask 0.1 --use_wandb --wandb_project "panopto" --wandb_name "${SCENE}_20_views" --mask_loss_from 2000 --checkpoint_iterations 100 1000 5000 10000 15000 20000 
    port=$((port+1))
done