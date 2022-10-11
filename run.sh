#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 main.py >> iron_log_mean_corp_regularize5.txt

# command to run: nohup bash run.sh &> no_hup.out &
