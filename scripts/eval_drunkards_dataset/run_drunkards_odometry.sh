#!/bin/bash

# Script to execute Drunkard's Odometry in all the four levels of difficulty of the Drunkard's Dataset test scenes.

for level_idx in 0 1 2 3
do	
        echo "###### Evaluating Drunkard's Odometry in scene $scene and level $level_idx ######"
	python scripts/eval_drunkards.py \
	--name="drunkards-odometry" \
	--datapath=/.../DrunkardsDataset320 \
	--res_factor=1. \
	--difficulty_level=$level_idx \
	--ckpt=/.../drunkards-odometry.pth \
	--test_scenes 0 4 5
done
