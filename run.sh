#!/bin/bash

export PYTHONUNBUFFERED=true
# CUDA_VISIBLE_DEVICES=1,0,2,3,5 python3 main.py >> iron_log_hb_concent_NTIntGrad_3class.txt
CUDA_VISIBLE_DEVICES=4,6,7,8,9 python3 main.py >> iron_log_hb_concent_medcam_3class.txt

# command to run: nohup bash run.sh &> no_hup.out &
