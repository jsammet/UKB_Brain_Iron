#!/bin/bash
export PYTHONUNBUFFERED=true
CUDA_VISIBLE_DEVICES=8 python3 linear_main.py >> check_linear_iron.txt

# command to run: nohup bash linear_run.sh &> lin_no_hup.out &
