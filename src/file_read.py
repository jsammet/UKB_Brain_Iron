import pandas as pd
# expect csv as input file

def file_reader(path):
	assert path.endswith('.csv')
	pd.read_csv(path)
	
