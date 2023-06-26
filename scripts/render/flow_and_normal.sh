#!/bin/bash

blender_exe="/.../blender-3.0.0-linux-x64/blender"
blend_files_path="/.../blender"
project_path="/.../DrunkardsOdometry"
output_dataset_path="/.../drunkards_dataset"

scenes="00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00018 00019"
levels="0 1 2 3"

# root path to the folder containing the blender project files. It should follow this structure:
# --> root
#     --> 00000
#         --> level0
#             --> workspace_flow.blend
#         ...
#     ...

for scene in $scenes
do
   for level_idx in $levels
   do
      level="level$level_idx"
      save_path="$output_dataset_path/$scene/$level"
      $blender_exe -b $blend_files_path/$scene/$level/workspace_flow.blend -P $project_path/scripts/render/flow_and_normal.py -- -o $save_path -s $scene
   done
done

