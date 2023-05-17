#!/bin/bash

blender_exe="$HOME/software/blender-3.0.0-linux-x64/blender"
blend_files_path="/media/david/DiscoDuroLinux/Datasets/drunk_blend_files"
project_path="/home/david/GitHub/DrunkardsOdometry"
output_dataset_path="/media/david/DiscoDuroLinux/Datasets/tmp"


scene="00000"
num_images=10
for level_idx in 1
do
   level="level$level_idx"
   pose_path="$output_dataset_path/$scene/$level/"
   $blender_exe -b $blend_files_path/$scene/$level/workspace.blend -P $project_path/scripts/render_pose.py -- -p $pose_path -n $num_images
done
