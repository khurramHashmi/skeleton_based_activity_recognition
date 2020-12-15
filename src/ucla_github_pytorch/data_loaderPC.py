from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

import random
import torch
import pickle
import h5py
import numpy as np

def get_data_list(data_path):
    f = h5py.File(data_path, 'r')
    data_list = []
    label_list = []
    for i in range(len(f['label'])):

        if np.shape(f[str(i)][:])[0] > 10:
            x = f[str(i)][:]
            # original matrix with probability
            y = f['label'][i]

            x = torch.tensor(x, dtype=torch.float)

            data_list.append(x)
            label_list.append(y)

    return data_list, label_list

def concate_data(data_path, seq_len = 10):
    data_list, label_list = get_data_list(data_path)

    feature_len = data_list[0].size()[-1]
    data = torch.tensor(())
    for i in range(len(label_list)):
        if data_list[i].size()[0] == seq_len:
            tmp = troch.flatten(data_list[i])
            data = torch.cat((data, tmp)).unsqueeze(0) 

        if data_list[i].size()[0] < seq_len:
          dif = seq_len - data_list.size()[0]
          tmp = torch.cat((data_list[i], torch.zeros((dif, feature_len))))
          tmp = torch.flatten(tmp)
          data = torch.cat((data, tmp)).unsqueeze(0) 
        
        if data_list[i].size()[0] > seq_len:
          tmp = data_list[i][:seq_len,:]
          tmp = torch.flatten(tmp).unsqueeze(0) 
          data = torch.cat((data, tmp))
    label_list = np.asarray(label_list)
    return data.numpy(), label_lists


def pad_collate(batch):
    lens = [x[1] for x in batch]

    data = [x[0] for x in batch]
    label = [x[2] for x in batch]
    label = np.asarray(label)

    # data = torch.tensor(data)
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    #print(lens,label)
    return xx_pad, lens, label




class MyAutoDataset(Dataset):
    def __init__(self, data, label):
      
        self.data = data
        self.label = label
        #self.xy = zip(self.data, self.label)


    def __getitem__(self, index):
        sequence = self.data[index, :]
        label = self.label[index]
        # Transform it to Tensor
        #x = torchvision.transforms.functional.to_tensor(sequence)
        #x = torch.tensor(sequence, dtype=torch.float)
        #y = torch.tensor([self.label[index]], dtype=torch.int)
        
        return sequence, label

    def __len__(self):
        return len(self.label)

class MyDataset(Dataset):
    def __init__(self, data_path):

        self.data, self.label = get_data_list(data_path)

        label = np.asarray(self.label)
        train_index = np.zeros(len(self.label))

    def __getitem__(self, index):
        sequence = self.data[index]
        label = self.label[index]

        return sequence, label

    def __len__(self):
        return len(self.label)

class NTU_Dataloader(Dataset):
    def __init__(self, data_path):

        with open(data_path, 'rb') as f:
            # self.sample_name, self.label = pickle.load(f, encoding='latin1')
            self.data = pickle.load(f)

        # label = np.asarray(self.label)
        # train_index = np.zeros(len(self.label))

    def __getitem__(self, index):
        sequence = self.data[index]['input']
        label = self.data[index]['label']
        sequence = torch.tensor(sequence)
        return sequence, label

    def __len__(self):
        return len(self.data)

class NTUDataset(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, pickle=False):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.pickle = pickle
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try: # new label pickle containing only the labels is now on second exception
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        except:
            try: # new label pickle containing only the labels
                with open(self.label_path, 'rb') as f:
                    self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f)

        # load data
        if self.pickle:
            self.data = pickle.load(self.data_path)
        elif self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path, allow_pickle=True)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if self.pickle:
            data_numpy = self.data[index]["input"]
        else:
            data_numpy = self.data[index]
        label = self.label[index]

        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map
        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        #Code FOR GET ITEM IN MS-SGF DATA STARTS HERE
        # data_numpy = np.array(data_numpy)
        # data_numpy = data_numpy[:, :, :, [0]]
        # data_numpy = np.moveaxis(data_numpy, [1, 0], [0, 2])
        # data_numpy = np.reshape(data_numpy, (300, 75))
        #
        # seq_length = len(data_numpy)
        #
        # # Reducing the size of the video by
        # # Pruning the sequence where repetition has started
        # check_count = 0
        # first_frame = data_numpy[0]
        #
        # for iter in range(len(data_numpy)):
        #     if iter != len(data_numpy) - 1 and check_count == 0:
        #         if (data_numpy[iter + 1] == first_frame).all():
        #             seq_length = iter
        #             check_count += 1
        #
        # #print(len(data_numpy))
        # #print(seq_length)
        # data_t = torch.tensor(data_numpy[:seq_length])
        # return data_t, seq_length, label
        #Code FOR GET ITEM IN MS-SGF DATA ENDS HERE


        #ANOTHER CODE FOR COMBINED SIT_STAND AVATAR FOR MSG3D DATA STARTS HERE
        # data_numpy = np.array(data_numpy)
        # data_numpy = np.reshape(data_numpy, (300, 75))
        #
        # # Reducing the size of the video by
        # # Pruning the sequence where repetition has started
        # check_count = 0
        # first_frame = data_numpy[0]
        # seq_length = 300
        #
        # for iter in range(len(data_numpy)):
        #     if iter != len(data_numpy) - 1 and check_count == 0:
        #         if (data_numpy[iter + 1] == first_frame).all():
        #             seq_length = iter
        #             check_count += 1
        #
        # #print(len(data_numpy))
        # #print(seq_length)
        # data_t = torch.tensor(data_numpy[:seq_length])
        # return data_t, seq_length, label
        #CODE FOR COMBINED SIT_STAND AVATAR FOR MSG3D DATA ENDS HERE



        #Code FOR GET ITEM FOR NOW SINCE SEQUENCE IS ALREADY REDUCED
        data_numpy = np.array(data_numpy).reshape(-1,75)
        seq_length = len(data_numpy)
        data_t = torch.tensor(data_numpy, dtype=torch.float32)

        return data_t, seq_length, label



    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
