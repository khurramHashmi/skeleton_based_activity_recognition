import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SkeletonsDataset(Dataset):
    """Skeletons dataset."""
    def __init__(self, data_file_info,batch_size):
        #Read the file containing the list of files having label and coordinates.
        self.data_files_info = pd.read_csv(data_file_info,delimiter="\t")
        self.batch_size = batch_size
        self.start=0

        remainder = (len(self.data_files_info) % batch_size)
        self.data_files_info = self.data_files_info[:len(self.data_files_info)-remainder]

        print(len(self.data_files_info))

    def __len__(self):
        # return self.chunk_size
        return len(self.data_files_info)


    def __getitem__(self, idx):
        # idx refers to the video number that should be retrieved
        # self.train_df = self.reader.get_chunk()
        file_names = []
        train_data_source = []
        labels = []

        # for index in range(self.start,self.start+self.batch_size):
        file_name = self.data_files_info.iloc[idx,0]
        file_names.append((file_name))
        batch_sample = pd.read_csv(file_name, delimiter="\t", header=None)

        # print(batch_sample)
        # label_list = batch_sample[0].replace("'", "").split(",")
        # label_list = label_list.split(",")

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
            data_skel.append(float(d))
            if count == 150:
                data_source.append(data_skel)
                data_skel = []
                count = 1
            else:
                count+=1

        # train_data_source.append()
        #normalizing the size of each video sequence with the number of skeleton
        # temp_vec = [0] * 100
        # for i in range(len(data_source),472):  # padding 0s vector to the maximum size available
        #     data_source.append(temp_vec)
        #
        # for i in range(len(data_source),100,-1):  # padding 0s vector to the maximum size available
        #     data_source.append(temp_vec)
        # print(" ORIGNAL LENGTH : ",str(len(data_source)))
        # Orignal Code


        #Now mamking embedding of skeleton sequence code starts here
        # data = self.train_df[1][idx].replace("'", "").split(",")
        # # data = data.split(",")
        # skel_embedding = dict()
        # data_source = []
        # data_skel = ""  # for each skeleton pose make it a string to consider as a key for skel_embedding dict
        # count = 1  # after 100 values, there is a new skeleton
        # skel_vec = 1  # Assiging value to each skeleton
        # rep_skel_vec =1
        #
        # for d in data:
        #     d = re.sub('\D', '', d)
        #     data_skel+=d
        #     if count == 100:
        #         if data_skel not in skel_embedding:
        #             skel_embedding[data_skel] = 60 + skel_vec # since there are 60 labels so values after that should be encoded
        #             data_source.append(60 + skel_vec)
        #             skel_vec += 1
        #         else:
        #             data_skel += str(rep_skel_vec)
        #             data_source.append(60 + skel_vec)
        #             rep_skel_vec += 1
        #             skel_vec += 1
        #             # print("SKELETON IS REPEATED!! \n VALUE CANNOT BE ENTERED INTO EMBEDDING")
        #         data_skel = ""
        #         count = 1
        #     else:
        #         count+=1
        # Now making embedding of skeleton sequence code ends here

        train_data_source = torch.tensor(data_source[0:100], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        # data_source = data_source.view(-1)

        # data_source = torch.Tensor(data_source).view(len(data_source), -1).t().contiguous()
        labels = labels.view(-1)

        self.start=self.batch_size
        return train_data_source, labels.t().contiguous(),file_names  #Taking Top 100 frames because of having extra data and unnecessary padding with 0s

