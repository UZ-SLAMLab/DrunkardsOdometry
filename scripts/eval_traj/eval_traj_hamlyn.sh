#!/bin/bash

root='/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn'

align="--align"
#align=""

#correct_scale="--correct_scale"
correct_scale=""

#align_origin="--align_origin"
align_origin=""

plot="--plot --plot_mode xy"
#plot=""

for scene in "test1" 
do    			
	#method="baseline_newGt_lr0.0002_10epochs_trainedInLevel1_epoch1"
	#method="baseline_newGt_lr0.0002_10epochs_trainedInLevel1_epoch8"
	method="baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1"

        echo "###### EVALUATING $method in SCENE $scene LEVEL $level_idx ######"
		
	echo "###### RPE traslation [m] (take the mean value) ######"
	evo_rpe tum "$root/$scene/$method/stamped_traj_estimate.txt" "$root/$scene/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation trans_part #--save_results "$root/$scene/level$level_idx/$method/results_rpe_tra.zip"

	echo "###### RPE rotation [º] (take the mean value) ######"
	evo_rpe tum "$root/$scene/$method/stamped_traj_estimate.txt" "$root/$scene/$method/stamped_traj_estimate.txt" $align $correct_scale $align_origin --pose_relation angle_deg #--save_results "$root/$scene/level$level_idx/$method/results_rpe_rot.zip"

	echo "###### ATE [m] (take the mean value) ######"
	evo_ape tum "$root/$scene/$method/stamped_traj_estimate.txt" "$root/$scene/$method/stamped_traj_estimate.txt" $align $correct_scale $plot $align_origin #--save_results "$root/$scene/level$level_idx/$method/results_ate.zip"
	
done

