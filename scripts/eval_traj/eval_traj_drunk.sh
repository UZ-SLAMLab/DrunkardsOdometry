#!/bin/bash

root='/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_drunk_current'

align="--align"
#align=""

#correct_scale="--correct_scale"
correct_scale=""

#align_origin="--align_origin"
align_origin=""

#plot="--plot --plot_mode xy"
plot=""

for level_idx in 0 1 2 3
do    	
	method="baseline_newGt_level$level_idx.epoch9"
	for scene in 00000 00004 00005
	do
        	echo "###### EVALUATING $method in SCENE $scene LEVEL $level_idx ######"
		echo "###### ATE [m] (take the mean value) ######"
		evo_ape tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $plot $align_origin #--save_results "$root/$scene/level$level_idx/$method/results_ate.zip"

		echo "###### RPE traslation [m] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation trans_part #--save_results "$root/$scene/level$level_idx/$method/results_rpe_tra.zip"

		echo "###### RPE rotation [º] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation angle_deg #--save_results "$root/$scene/level$level_idx/$method/results_rpe_rot.zip"
	done
done



align="--align"
#align=""

#correct_scale="--correct_scale"
correct_scale=""

#align_origin="--align_origin"
align_origin=""

#plot="--plot --plot_mode xy"
plot=""

for level_idx in 1
do    	
	method="baseline_newGt_back_level$level_idx.epoch9"
	for scene in 00000 00004 00005
	do
        	echo "###### EVALUATING $method in SCENE $scene LEVEL $level_idx ######"
		echo "###### ATE [m] (take the mean value) ######"
		evo_ape tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $plot $align_origin #--save_results "$root/$scene/level$level_idx/$method/results_ate.zip"

		echo "###### RPE traslation [m] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation trans_part #--save_results "$root/$scene/level$level_idx/$method/results_rpe_tra.zip"

		echo "###### RPE rotation [º] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation angle_deg #--save_results "$root/$scene/level$level_idx/$method/results_rpe_rot.zip"
	done
done



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
        	echo "###### EVALUATING $method in SCENE $scene LEVEL $level_idx ######"		

		echo "###### RPE traslation [m] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation trans_part #--save_results "$root/$scene/level$level_idx/$method/results_rpe_tra.zip"

		echo "###### RPE rotation [º] (take the mean value) ######"
		evo_rpe tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation angle_deg #--save_results "$root/$scene/level$level_idx/$method/results_rpe_rot.zip"

		echo "###### ATE [m] (take the mean value) ######"
		evo_ape tum "$root/$scene/level$level_idx/$method/pose.txt" "$root/$scene/level$level_idx/$method/stamped_traj_estimate.txt" $align $correct_scale $plot $align_origin #--save_results "$root/$scene/level$level_idx/$method/results_ate.zip"
	done
done
