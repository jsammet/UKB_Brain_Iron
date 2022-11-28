#!/bin/bash
export PYTHONUNBUFFERED=true
CUDA_VISIBLE_DEVICES=4,5 python3 image_averaging.py >> check_avg.txt

# command to run: nohup bash linear_run.sh &> lin_no_hup.out &
