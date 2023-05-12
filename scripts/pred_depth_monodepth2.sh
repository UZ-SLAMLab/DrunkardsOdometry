#!/bin/bash

script_path="/home/david/GitHub/endodepth/monodepth2-master"
model_path="/home/david/GitHub/monodepth2/models/mono_1024x320"

echo "Loading model from $script_path"

parent_image_path="/media/david/ExtDisk1/drunkSLAM/drunk_dataset_6"
parent_output_directory="/media/david/ExtDisk1/drunkSLAM/edam_depthByMono1024x320_1_6th"


for level_idx in 0 1 2 3
do    		
	for scene in 00000 00004 00005
	do
		image_path="$parent_image_path/$scene/level$level_idx/color"
		output_directory="$parent_output_directory/$scene/level$level_idx/depth"
        	echo "###### Predicting depth in SCENE $scene LEVEL $level_idx ######"		
		echo "Image color input path: $image_path"
		echo "Depth output path: $output_path"

		python $script_path/test_simple_drunk.py --model_path $model_path --image_path $image_path --output_directory $output_directory
		
	done
done







