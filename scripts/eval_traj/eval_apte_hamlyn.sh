#!/bin/bash

script_path="/home/david/GitHub/RAFT-3D_def/scripts/eval_apte.py"
pose_forward="/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/test1/edam/stamped_traj_estimate.txt"
pose_backward="/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/test1_backward/edam/stamped_traj_estimate.txt"
save_folder="/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/test1/edam"


for scene in "test1" "test1_backward" "test17" "test17_backward" "test22" "test22_backward"
do
        echo "###### Evaluating APTE rel scene $scene ######"
	python $script_path --pose_forward=$pose_forward --pose_backward=$pose_backward --save_folder=$save_folder
done

