#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py >> iron_log_hb_mean_corp_lowLR_2class.txt


# command to run: nohup bash run.sh &> no_hup.out &
