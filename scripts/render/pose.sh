#!/bin/bash

blender_exe="/.../blender-3.0.0-linux-x64/blender"
blend_files_path="/.../drunkards_blend_files"
project_path="/.../DrunkardsOdometry"
output_dataset_path="/.../drunkards_dataset"

scenes="00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00018 00019"
levels="0 1 2 3"

for scene in $scenes
do
   for level_idx in $levels
   do
      level="level$level_idx"
      pose_path="$output_dataset_path/$scene/$level"
      $blender_exe -b $blend_files_path/$scene/$level/workspace.blend -P $project_path/scripts/render/pose_matrix.py -- -p $pose_path -s $scene
      python $project_path/scripts/render/matrix_to_quat_pose.py -i $pose_path -o $pose_path
      rm $pose_path/pose_matrix.txt
   done
done
