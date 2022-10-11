#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python3 linear_main.py >> linear_iron_check.txt

# command to run: nohup bash linear_run.sh &> lin_no_hup.out &
