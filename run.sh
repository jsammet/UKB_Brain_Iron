#!/bin/bash

export PYTHONUNBUFFERED=true

echo "hb 10"
CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_hb_concent_final100_10class.txt
# echo "hb 50 no batch"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_hb_concent_no_batch_final100_50class.txt
# echo "flip hb 10"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_final_flip_patience12_10class.txt

# echo "std low hb 3"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_images_low_stddev_3class.txt

# echo "hct 3"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_hct_perc_final60_3class.txt
# echo "hct 3 no batch"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_hct_perc_no_batch_final60_3class.txt
# echo "hct 10 no batch"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_hct_perc_no_batch_final100_10class.txt
# echo "mean corp 3"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_mean_corp_final60_3class.txt
# echo "mean corp 10 no batch"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_mean_corp_no_batch_final10_10class.txt
# echo "mean corp 10"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_mean_corp_NTsqIntGrad_10class.txt
# echo "hct 10"
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_hct_perc_NTsqIntGrad_10class.txt

# echo "hb 3 no batch sal map"
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_hb_concent_no_batch_final60_3class.txt

# command to run: nohup bash run.sh &> no_hup.out &
# check for changes: class, epochs, iron_measure, sal_map (flip or not)
