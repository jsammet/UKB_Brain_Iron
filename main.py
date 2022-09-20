import os

import numpy as np
import pandas as pd

import src.model
import src.dataloader
import src.file_read as file_read

input_path='final_brain_vil_info.csv'

input_file=file_read(input_path)

dataloader=src.dataloader(input_file)

