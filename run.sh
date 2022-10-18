#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py >> iron_log_mean_corp_bigLR_weight_3class.txt


# command to run: nohup bash run.sh &> no_hup.out &
