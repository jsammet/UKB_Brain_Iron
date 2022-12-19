#!/bin/bash
export PYTHONUNBUFFERED=true
python3 image_averaging.py >> check_avg.txt

# command to run: nohup bash img_avg_run.sh &> no_hup_avg.out &
