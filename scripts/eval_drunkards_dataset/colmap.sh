#!/bin/bash

# Script to evaluate Colmap in all the four levels of difficulty of the Drunkard's Dataset test scenes.
# As Colmap only uses the RGB images, scale alignment with SIM(3) is needed.

root='/.../evaluations_drunkards_dataset'
# root path to the folder containing the poses of all methods. It should follow this structure:
# --> root
#     --> 00000
#         --> level0
#             --> colmap
#                 --> pose.txt       Ground truth camera trajectory poses
#                 --> pose_est.txt   Estimated camera trajectory poses
#         ...
#     ...

align="--align"
#align=""

correct_scale="--correct_scale"
#correct_scale=""

#align_origin="--align_origin"
align_origin=""

#plot="--plot --plot_mode xy"
plot=""

for level_idx in 0 1 2 3
do    	
	method="colmap"
	for scene in 00000 00004 00005
	do
        	echo "###### Evaluating $method in scene $scene and level $level_idx ######"

		echo "###### RPE traslation [m] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/pose_est.txt" $align $correct_scale $align_origin --pose_relation trans_part --save_results "$root/$scene/level$level_idx/$method/results_rpe_tra.zip"

		echo "###### RPE rotation [º] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/pose_est.txt" $align $correct_scale $align_origin --pose_relation angle_deg --save_results "$root/$scene/level$level_idx/$method/results_rpe_rot.zip"

		echo "###### ATE [m] (take the mean value) ######"
		evo_ape tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/pose_est.txt" $align $correct_scale $plot $align_origin --save_results "$root/$scene/level$level_idx/$method/results_ate.zip"
	done
done
