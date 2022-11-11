#!/bin/bash

export PYTHONUNBUFFERED=true
CUDA_VISIBLE_DEVICES=2,3,8,9 python3 main.py >> iron_log_hb_concent_GradCAM_3class.txt


# command to run: nohup bash run.sh &> no_hup.out &
