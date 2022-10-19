#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py >> iron_log_hb_concent_corLoss_5class.txt


# command to run: nohup bash run.sh &> no_hup.out &
