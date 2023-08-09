#!/bin/bash

# Script to compute the SIM(3) scale factor to align the estimated trajectories of Drunkard's Odometry and DROID-SLAM
# to the estimated trajectory of Endo-Depth-and-Motion.
# To compute the APTE of all the methods, introduce these scale factors in eval_apte.py and execute it.

root='/.../evaluations_hamlyn'

align="--align"
#align=""

correct_scale="--correct_scale"
#correct_scale=""

#align_origin="--align_origin"
align_origin=""

#plot="--plot --plot_mode xy"
plot=""

reference_method=edam

for scene in "test1" "test17"
do    	
	for method in  "drunkards-odometry" "droidslam"
	do
        	echo "###### Comparing $reference_method and $method in scene $scene ######"	
	
		evo_ape tum "$root/$scene/$reference_method/pose_est_total.txt" "$root/$scene/$method/pose_est_total.txt" $align $correct_scale $plot $align_origin -va --save_results "$root/$scene/level$level_idx/$method/results_ate.zip"
	done
	
done

