import re
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from ntu_preprocess import *
import cv2
from torch.utils.data import Dataset, DataLoader


class FolderDataset(Dataset):
    """Skeletons dataset."""

    def __init__(self, path):

        self.base_path = path
        self.features = os.listdir(path)
        print('Total examples: ', len(self.features))
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        data_source = pickle.load(open(os.path.join(self.base_path, self.features[idx]), 'rb'))
        in_f = torch.tensor(data_source[:, :, :3], dtype=torch.float)
        label = self.features[idx].split('_')[-1].split('.')[0]
        labels = torch.tensor([int(label)], dtype=torch.long)
        in_f = in_f.permute(2, 0, 1) # 180 * 180 * 6 -> 6 * 180 * 180
        return in_f, labels

class SimpleDataset(Dataset):
    """Skeletons dataset."""

    def __init__(self, path, train=True):

        if train:
            self.features = pickle.load(open(os.path.join(path, 'train_features.p'), 'rb'))
            self.labels = pickle.load(open(os.path.join(path, 'lab.p'), 'rb'))
        else:
            self.features = pickle.load(open(os.path.join(path, 'test_features.p'), 'rb'))
            self.labels = pickle.load(open(os.path.join(path, 'test_lab.p'), 'rb'))

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):

        data_source = self.features[idx]# [np.newaxis, :]
        in_f = torch.tensor(data_source, dtype=torch.float)
        labels = torch.tensor([np.argmax(self.labels[idx])], dtype=torch.long)

        return in_f, labels


class SkeletonsDataset(Dataset):
    """Skeletons dataset."""
    # def __init__(self, data_file_info = None,batch_size, base_path, pickle_ptah_val = None, image_dataset=False):

    def __init__(self, base_path, image_dataset=False, mode='train'):

        self.input_features = pd.read_csv(base_path)


        # tr_path = "trans_train_data.pkl"
        # te_path = "trans_test_data.pkl"
        #
        # # Preprocess the dataset(NTU here for demo) to generate data for
        # pickle_path = './pickles'
        # if not os.path.exists(pickle_path):
        #     os.mkdir(pickle_path)
        #     print('Generating files')
        #
        #     dsamp_train, dsamp_test, \
        #     fea, lab, seq_len_new, \
        #     test_fea, test_lab, test_seq_len_new = preprocess_pipeline(base_path, tr_path, te_path,
        #                                                                mode="cross_subject_data",
        #                                                                dsamp_frame=50)
        #
        #     pickle.dump(dsamp_train, open(pickle_path + "/dsamp_train.p", "wb"))
        #     pickle.dump(dsamp_test, open(pickle_path + "/dsamp_test.p", "wb"))
        #
        #     self.input_features = dsamp_train
        #     self.labels = lab
        #     pickle.dump(fea, open(pickle_path + "/fea.p", "wb"))
        #     pickle.dump(lab, open(pickle_path + "/lab.p", "wb"))
        #     pickle.dump(seq_len_new, open(pickle_path + "/seq_len_new.p", "wb"))
        #     pickle.dump(test_fea, open(pickle_path + "/test_fea.p", "wb"))
        #     pickle.dump(test_lab, open(pickle_path + "/test_lab.p", "wb"))
        #
        #     # pickle.dump(self.means, open(pickle_path + "/means.p", "wb"))
        #     # pickle.dump(self.std_devs, open(pickle_path + "/std_devs.p", "wb"))
        #
        #     # pickle.dump(test_seq_len_new, open(pickle_path + "/test_seq_len_new.p", "wb"))
        #     del dsamp_test
        #     del test_lab
        # else:
        #     print('Loading files')
        #     if mode == 'train':
        #         self.input_features = pickle.load(open(pickle_path + "/dsamp_train.p", "rb"))
        #
        #         self.labels = pickle.load(open(pickle_path + "/lab.p", "rb"))
        #     else:
        #         self.input_features = pickle.load(open(pickle_path + "/dsamp_test.p", "rb"))
        #         self.labels = pickle.load(open(pickle_path + "/test_lab.p", "rb"))
        #     # self.means = pickle.load(open(pickle_path + "/means.p", "rb"))
        #     # self.std_devs = pickle.load(open(pickle_path + "/std_devs.p", "rb"))
        #
        #     # fea = pickle.load(open(pickle_path + "/fea.p", "rb"))
        #
        #     # seq_len_new = pickle.load(open(pickle_path + "/seq_len_new.p", "rb"))
        #     # test_fea = pickle.load(open(pickle_path + "/test_fea.p", "rb"))
        #     # test_lab = pickle.load(open(pickle_path + "/test_lab.p", "rb"))
        #     # test_seq_len_new = pickle.load(open(pickle_path + "/test_seq_len_new.p", "rb"))
        #
        # #Read the file containing the list of files having label and coordinates.
        # # print(self.means.shape)
        # # print(self.std_devs.shape)
        # self.print_once=True
        # # # removing the troubled classes for now code starts here
        # # self.data_files_info = self.remove_troubled_classes()
        # # # removing the troubled classes for now code Ends here
        # # self.batch_size = batch_size
        # # remainder = (len(self.data_files_info) % batch_size)
        # # self.data_files_info = self.data_files_info[:len(self.data_files_info)-remainder]
        # #
        # # self.mean_std = pickle.load(open(pickle_path_train, 'rb'))
        # # temp_mean = pickle.load(open(pickle_path_val, 'rb'))
        # # self.mean_std['mean'] = (self.mean_std['mean'] + temp_mean['mean'])/2
        # # self.mean_std['std'] = (self.mean_std['std'] + temp_mean['std'])/2
        # print("***** Example shape {} *****".format(len(self.input_features)))
        self.image_dataset=image_dataset

    def __len__(self):
        # return self.chunk_size
        return len(self.input_features)


    def __getitem__(self, idx):

        data_source = cv2.imread(self.input_features.iloc[idx, 0])
        label = int(self.input_features.iloc[idx, 0].split('_')[-1].split('.')[0])

        if not self.image_dataset:
            # reshape for seq
            data_source = data_source.reshape((100, 300))
        # data_source = self.input_features[idx]
        # data_source = np.array(data_source)
        #
        # if self.image_dataset: # for creating image dataset
        #     # reshape the values into image form
        #     inputs = np.zeros((50, 75), dtype=float)
        #     inputs[:data_source.shape[0], :] = np.copy(data_source)
        #     data_source = inputs.reshape((50, 25, 3))
        #     assert data_source.shape == (50,25,3), print('wrong dimension image created')
        #
        #     if self.print_once:
        #         print('Data shape {}'.format(data_source.shape))
        #         self.print_once=False
        #
        # # data_source -= self.means
        # # data_source /= self.std_devs
        train_data_source = torch.tensor(data_source, dtype=torch.float)
        train_data_source = train_data_source.permute(2, 0, 1) if self.image_dataset else train_data_source
        temp_labels = []
        #
        # temp_labels.append(np.argmax(self.labels[idx]))
        #
        labels = torch.tensor([label], dtype=torch.long)

        return train_data_source, labels
