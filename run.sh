#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,7,8 python3 main.py >> iron_log_mean_corp_50class.txt

# command to run: nohup bash run.sh &> no_hup.out &
