#!/bin/bash

script_path="/.../DrunkardsOdometry"


echo "###### Drunkard's Odometry ######"
python scripts/eval_hamlyn.py \
--datapath=/home/david/datasets/hamlyn_for_drunk_paper \
--ckpt=/home/david/GitHub/RAFT-3D_def/checkpoints/cluster/baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36/000009.pth \
--gpus=1 \
--test_scenes \
test1 \
test1_backward \
test17 \
test17_backward \
--save_path=/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn \
--estimate_pose_updates \
--makeGradientClip \
--makePoseGradientClip \
--motion_info_clamp 




