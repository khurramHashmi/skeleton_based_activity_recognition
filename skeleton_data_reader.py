import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader

class SkeletonsDataset(Dataset):
    """Skeletons dataset."""
    def __init__(self, data_file):
        self.train_df = pd.read_csv(data_file,delimiter="\t", header=None)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # idx refers to the video number that should be retrieved
        label_list = self.train_df[0][idx].replace("'", "").split(",")
        # label_list = label_list.split(",")

        # converting the labels from string to a vectors
        labels = []
        for label in label_list:
            label = re.sub('\D', '', label)
            # for i in range(100):
            labels.append(int(label))

        # parsing string data as a list of vectors
        # data = self.train_df[1][idx].replace("'", "").split(",")
        # # data = data.split(",")
        # data_source = []
        # data_skel=[] # for each skeleton pose
        # count = 1
        # for d in data:
        #     d = re.sub('\D', '', d)
        #     data_skel.append(int(d))
        #     if count == 100:
        #         data_source.append(data_skel)
        #         data_skel = []
        #         count = 1
        #     else:
        #         count+=1
        # print(" ORIGNAL LENGTH : ",str(len(data_source)))
        # Orignal Code


        #Now mamking embedding of skeleton sequence code starts here
        data = self.train_df[1][idx].replace("'", "").split(",")
        # data = data.split(",")
        skel_embedding = dict()
        data_source = []
        data_skel = ""  # for each skeleton pose make it a string to consider as a key for skel_embedding dict
        count = 1  # after 100 values, there is a new skeleton
        skel_vec = 1  # Assiging value to each skeleton
        rep_skel_vec =1

        for d in data:
            d = re.sub('\D', '', d)
            data_skel+=d
            if count == 100:
                if data_skel not in skel_embedding:
                    skel_embedding[data_skel] = 60 + skel_vec # since there are 60 labels so values after that should be encoded
                    data_source.append(60 + skel_vec)
                    skel_vec += 1
                else:
                    data_skel += str(rep_skel_vec)
                    data_source.append(60 + skel_vec)
                    rep_skel_vec += 1
                    skel_vec += 1
                    # print("SKELETON IS REPEATED!! \n VALUE CANNOT BE ENTERED INTO EMBEDDING")
                data_skel = ""
                count = 1
            else:
                count+=1
        # Now making embedding of skeleton sequence code ends here

        data_source = torch.tensor(data_source, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        # data_source = data_source.view(-1)

        # data_source = torch.Tensor(data_source).view(len(data_source), -1).t().contiguous()
        labels = labels.view(-1)

        return data_source, labels.t().contiguous()

