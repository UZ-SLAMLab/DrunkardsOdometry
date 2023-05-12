#!/bin/bash

root='/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn'

align="--align"
#align=""

correct_scale="--correct_scale"
#correct_scale=""

#align_origin="--align_origin"
align_origin=""

#plot="--plot --plot_mode xy"
#plot=""

reference_method=edam

for scene in "test1" "test17" "test22"
do    	
	for method in "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3_fr2" "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch3_fr2" "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch9_fr2" "droidslam" "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-3_trainedInLevel1_epoch9" "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1" "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3"
	do

        	echo "###### Comparing $reference_method and $method in scene $scene ######"	
	
		evo_ape tum "$root/$scene/$reference_method/stamped_traj_estimate_total.txt" "$root/$scene/$method/stamped_traj_estimate_total.txt" $align $correct_scale $plot $align_origin -va #--save_results "$root/$scene/level$level_idx/$method/results_ate.zip"
	done
	
done

