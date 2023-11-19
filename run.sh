#!/bin/bash

export PYTHONUNBUFFERED=true

#### 3 CLASS NO BATCH MODEL MAPS
echo "hb 3 no_batch maps"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python3 main.py --iron hb_concent --class_nb 3 --eps 100 --batch no_batch --no-augment --no-create_maps --seed 44 >> iron_log_hb_concent_60eps_3class.txt
# echo "hb 10 no_batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 10 --eps 100 --batch no_batch --no-augment --no-create_maps --seed 1337 >> iron_log_hb_concent_100eps_10class.txt

#### 3 CLASS BATCH MODEL MAPS
# echo "hb 3 batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7,8,9 python3 main.py --iron hb_concent --class_nb 3 --eps 60 --batch batch --no-augment --no-create_maps --seed 1337 >> iron_log_hb_concent_batch_60eps_3class.txt
#### 10 CLASS BATCH MODEL MAPS
# echo "hb 10 batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 10 --eps 100 --batch batch --no-augment --no-create_maps --seed 1337 >> iron_log_hb_concent_batch_100eps_10class.txt
#### 20 CLASS BATCH MODEL MAPS
# echo "hb 20 batch maps"
#CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 20 --eps 100 --batch batch --no-augment --no-create_maps --seed 1337 >> iron_log_hb_concent_batch_100eps_20class.txt
#### 35 CLASS BATCH MODEL MAPS
# echo "hb 35 batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 35 --eps 200 --batch batch --no-augment --no-create_maps --seed 1337 >> iron_log_hb_concent_batch_200eps_35class.txt

#### 3 CLASS AUGMENTATION BATCH MODEL MAPS
# echo "hb 3 augment batch maps"
# CUDA_VISIBLE_DEVICES=6,7,8,9 python3 main.py --iron hb_concent --class_nb 3 --eps 200 --batch batch --augment --no-create_maps --seed 1337 >> iron_log_hb_concent_augment_batch_200eps_3class.txt
#### 10 CLASS AUGMENTATION BATCH MODEL MAPS
# echo "hb 10 augment batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 10 --eps 300 --batch batch --augment --no-create_maps --seed 1337 >> iron_log_hb_concent_augment_batch_300eps_10class.txt
#### 20 CLASS AUGMENTATION BATCH MODEL MAPS
# echo "hb 20 augment batch maps"
# CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py --iron hb_concent --class_nb 20 --eps 300 --batch batch --augment --no-create_maps --seed 1337 >> iron_log_hb_concent_augment_batch_300eps_20class.txt

# command to run: nohup bash run.sh &> no_hup.out &
# check for changes: iron measure, class number, epoch number, batch normalization, augmentation of training images, create attention maps
