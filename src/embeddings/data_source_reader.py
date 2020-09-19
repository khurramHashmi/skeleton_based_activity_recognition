import re
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

class SkeletonsDataset(Dataset):
    """Skeletons dataset."""
    def __init__(self, data_file_info,batch_size, pickle_path_train, pickle_path_val):
        #Read the file containing the list of files having label and coordinates.
        self.data_files_info = pd.read_csv(data_file_info,delimiter=",", index_col=False)
        self.batch_size = batch_size

        remainder = (len(self.data_files_info) % batch_size)
        self.data_files_info = self.data_files_info[:len(self.data_files_info)-remainder]

        self.mean_std = pickle.load(open(pickle_path_train, 'rb'))
        temp_mean = pickle.load(open(pickle_path_val, 'rb'))
        self.mean_std['mean'] = (self.mean_std['mean'] + temp_mean['mean'])/2
        self.mean_std['std'] = (self.mean_std['std'] + temp_mean['std'])/2

    def __len__(self):
        # return self.chunk_size
        return len(self.data_files_info)


    def __getitem__(self, idx):
        # idx refers to the video number that should be retrieved
        # self.train_df = self.reader.get_chunk()
        file_names = []
        labels = []

        # for index in range(self.start,self.start+self.batch_size):
        file_name = self.data_files_info.iloc[idx,1]
        
        file_names.append((file_name))
        batch_sample = pd.read_csv(file_name, delimiter="\t", header=None)

        # converting the labels from string to a vectors
        list_str=batch_sample[0].tolist()[0]
        label_list = list_str.replace("'", "").split(",")
        for label in label_list:
            label = re.sub('\D', '', label)
            # for i in range(100):
        labels.append(int(label)-1)

        # parsing string data as a list of vectors
        data = batch_sample[1].tolist()[0]
        data = data.replace("'", "").split(",")
        # data = data.split(",")
        data_source = []
        data_skel=[] # for each skeleton pose
        count = 1
        for d in data:
            d = re.sub('\D', '', d)
            # print(type(d))
            data_skel.append(int(d))
            if count == 100:
                data_source.append(data_skel)
                data_skel = []
                count = 1
            else:
                count+=1
        data_source = np.array(data_source)
        data_source = (data_source - self.mean_std['mean'])/self.mean_std['std']
        train_data_source = torch.tensor(data_source, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        labels = labels.view(-1)
        # return train_data_source, labels.t().contiguous(),file_names  #Taking Top 100 frames because of having extra data and unnecessary padding with 0s

        return train_data_source, labels.t().contiguous(), file_names  # Taking Top 100 frames because of having extra data and unnecessary padding with 0s

