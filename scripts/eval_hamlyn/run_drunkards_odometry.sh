#!/bin/bash

# Script to execute Drunkard's Odometry in the test scenes of the Hamlyn Dataset.

echo "###### Evaluating Drunkard's Odometry with deformation (trained in level 1 of the Drunkard's Dataset) in Hamlyn Dataset ######"
python scripts/eval_hamlyn.py \
--name=drunkards-odometry-hamlyn-w-deformation \
--datapath=/.../HamlynData \
--ckpt=/.../drunkards-odometry-hamlyn-w-deformation.pth \
--test_scenes \
test1 \
test1_backward \
test17 \
test17_backward


echo "###### Evaluating Drunkard's Odometry without deformation (trained in level 0 of the Drunkard's Dataset) in Hamlyn Dataset ######"
python scripts/eval_hamlyn.py \
--name=drunkards-odometry-hamlyn-wo-deformation \
--datapath=/.../HamlynData \
--ckpt=/.../drunkards-odometry-hamlyn-wo-deformation.pth \
--test_scenes \
test1 \
test1_backward \
test17 \
test17_backward


