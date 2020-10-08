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
        self.data_files_info = pd.read_csv(data_file_info,delimiter=",")

        # removing the troubled classes for now code starts here
        # self.data_files_info = self.remove_troubled_classes()
        # removing the troubled classes for now code Ends here

        self.batch_size = batch_size

        remainder = (len(self.data_files_info) % batch_size)
        self.data_files_info = self.data_files_info[:len(self.data_files_info)-remainder]

        # self.mean_std = pickle.load(open(pickle_path_train, 'rb'))
        # temp_mean = pickle.load(open(pickle_path_val, 'rb'))
        # self.mean_std['mean'] = (self.mean_std['mean'] + temp_mean['mean'])/2
        # self.mean_std['std'] = (self.mean_std['std'] + temp_mean['std'])/2
        print("***** Example shape {} *****".format(self.data_files_info.shape))

    def __len__(self):
        # return self.chunk_size
        return len(self.data_files_info)


    def __getitem__(self, idx):
        # idx refers to the video number that should be retrieved
        # self.train_df = self.reader.get_chunk()
        file_names = []
        labels = []

        # for index in range(self.start,self.start+self.batch_size):
        file_name = self.data_files_info.iloc[idx, 0]

        file_names.append((file_name))
        # batch_sample =  np.load(file_name, allow_pickle=True).item()
        # batch_sample_temp = pickle.load(open(self.temp.iloc[idx,0], 'rb'))
        batch_sample = pd.read_csv(file_name, delimiter="\t", header=None)

        # converting the labels from string to a vectors
        list_str=batch_sample[0].tolist()[0]
        label_list = list_str.replace("'", "").split(",")
        for label in label_list:
            label = re.sub('\D', '', label)
            # for i in range(60):

        # labels.append(batch_sample['labels'][0]-1)

        labels.append(int(label) - 1)

        # #Only for deleting 3 classes rn
        # label = int(label)
        # if label > 10:
        #     label -=3
        # # Only for deleting 3 classes rn
        # labels[lab_idx] = labels[lab_idx][:60]

        # parsing string data as a list of vectors
        data = batch_sample[1].tolist()[0]
        data = data.replace("'", "").split(",")
        # data = data.split(",")
        data_source = []
        data_skel=[] # for each skeleton pose
        count = 1
        for d in data:
            d = d.replace("[","")
            d = d.replace("]", "")
            data_skel.append(float(d))
            if count == 150: # replace this with variable
                data_source.append(data_skel)
                data_skel = []
                count = 1
            else:
                count+=1
        # print(type(data_source[0][0]), type(batch_sample_temp['values'][0][0]))
        # data_source = batch_sample['values']

        # data_source = np.array()

        # data_source = (data_source - self.mean_std['mean'])/self.mean_std['std']
        train_data_source = torch.tensor(data_source, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        labels = labels.view(-1)
        #print(train_data_source.shape, labels.shape)
        # return train_data_source, labels.t().contiguous(),file_names  #Taking Top 100 frames because of having extra data and unnecessary padding with 0s
        return train_data_source, labels.t().contiguous(), file_names  # Taking Top 100 frames because of having extra data and unnecessary padding with 0s

    def remove_troubled_classes(self):
        print("ORIGNAL SIZE : ",len(self.data_files_info))
        remove_count,idx = 0,0
        while True:

            file_name = self.data_files_info.iloc[idx, 0]
            label = int(file_name.split("A")[1].split(".")[0])

            if label == 10 or label == 11 or label == 12 or label == 16 or label == 17 or label == 29 or label == 41 or label == 44 or label == 47:
                self.data_files_info = self.data_files_info.drop(idx)
                remove_count += 1
            if idx >= (len(self.data_files_info) - 1):
                break
            idx += 1
        return self.data_files_info
