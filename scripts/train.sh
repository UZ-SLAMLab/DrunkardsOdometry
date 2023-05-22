#!/bin/bash

python scripts/train.py \
--name=drunkards-odometry-test \
--datapath=/home/david/GitHub/DrunkardsOdometry/datasets/DrunkardsDatasetSample \
--difficulty_level=1 \
--depth_augmentor \
--res_factor=3.2 \
--batch_size=2 \
--train_scenes 0 \
--val_scenes 0 
