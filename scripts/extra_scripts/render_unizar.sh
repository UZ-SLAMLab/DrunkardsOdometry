#!/bin/bash

blender_exe="$HOME/Software/blender-3.2.0-linux-x64/blender"
dataset_folder="/media/recasens/ExtDisk1/drunkSLAM/drunk_dataset"
render_scene_file="/media/recasens/ExtDisk1/drunkSLAM/scripts/render_scene.py"


scene="00009"
num_images=8098
for level_idx in 0 1 2 3
do
   level="level$level_idx"
   pose_folder="$dataset_folder/$scene/$level/pose/"
   color_folder="$dataset_folder/$scene/$level/color/"
   $blender_exe -b $dataset_folder/$scene/$level/workspace.blend -P $render_scene_file -- -p $pose_folder -c $color_folder -n $num_images
done
