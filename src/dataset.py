import os
import pandas as pd
import nibabel as nib

class swi_dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_path,label_path):
        'Initialization'
        self.label_list=pd.read_csv(label_path)
        self.img_dir=img_path
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Create image path and read image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = nib.load(img_path)
        # Create label information accordingly
        label = self.img_labels.iloc[idx, 1]

        #return both together
        return image, label
