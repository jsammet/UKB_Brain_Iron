#!/bin/bash

export PYTHONUNBUFFERED=true

# echo "hb 3 no_batch maps"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 3 --eps 60 --batch no_batch --augment False --create_maps True >> iron_log_hb_concent_60eps_3class.txt
# echo "hb 10 no_batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 10 --eps 100 --batch no_batch --augment False --create_maps False >> iron_log_hb_concent_100eps_3class.txt
# echo "hb 20 no_batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 20 --eps 100 --batch no_batch --augment False --create_maps False >> iron_log_hb_concent_100eps_3class.txt

# echo "hb 3 batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 3 --eps 60 --batch batch --augment False --create_maps True >> iron_log_hb_concent_batch_60eps_3class.txt
# echo "hb 10 batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 10 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt
# echo "hb 20 batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 20 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt
# echo "hb 50 batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py --iron hb_concent --class_nb 50 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt

# echo "hb 3 batch flip"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py --iron hb_concent --class_nb 3 --eps 60 --batch batch --augment False --create_maps True >> iron_log_hb_concent_batch_60eps_3class.txt
# echo "hb 10 batch flip"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py --iron hb_concent --class_nb 10 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt
# echo "hb 20 batch flip"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py --iron hb_concent --class_nb 20 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt
# echo "hb 50 batch flip"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py --iron hb_concent --class_nb 50 --eps 100 --batch batch --augment False --create_maps False >> iron_log_hb_concent_batch_100eps_3class.txt

# command to run: nohup bash run.sh &> no_hup.out &
# check for changes: class, epochs, model, iron_measure, sal_map (flip or not)
