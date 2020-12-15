from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from io import open
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from PCNet import *
from utilitiesPC import *
from data_loaderPC import *
from SV_train import *


root = '/home/hashmi/Desktop/activity_recognition/Predict-Cluster/ucla_github_pytorch/'

## training procedure
teacher_force = False
fix_weight = True
fix_state = False

if fix_weight:
    network = 'FW'

if fix_state:
    network = 'FS'

if not fix_state and not fix_weight:
    network = 'O'

# hyperparameter
feature_length = 75
hidden_size =1024
batch_size = 64
en_num_layers = 3
de_num_layers = 1
print_every = 1
learning_rate = 0.001
epoch = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global variable
ProjectFolderName = './UCLAdata/'
root_path = "/home/hashmi/Desktop/dataset/activity_recognition/NTURGBD-60_120/ntu_60/cross_subject_data/"
# root_path = "/home/hashmi/Desktop/dataset/activity_recognition/ntu/xsub/"
data_path_train = root_path + 'trans_train_avatar.npy'
data_path_train_label = root_path + 'train_label.pkl'

# dataset_train = NTU_Dataloader("/home/hashmi/Desktop/dataset/activity_recognition/NTURGBD-60_120/ntu_60/cross_subject_data/trans_train_data.pkl")

data_path_test = root_path + 'trans_test_avatar.npy' #test_trans_pkl.npy
data_path_test_label = root_path + 'val_label.pkl'

# dataset_test = NTU_Dataloader("/home/hashmi/Desktop/dataset/activity_recognition/NTURGBD-60_120/ntu_60/cross_subject_data/trans_test_data.pkl")

dataset_train = NTUDataset(data_path_train, data_path_train_label,use_mmap=False)
dataset_test = NTUDataset(data_path_test, data_path_test_label, use_mmap=False)
shuffle_dataset = True
dataset_size_train = len(dataset_train)
dataset_size_test = len(dataset_test)

indices_train = list(range(dataset_size_train))
indices_test = list(range(dataset_size_test))
batch_size = 32
random_seed = 11111
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)

print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_test)))
# seperate train and validation
train_sampler = SubsetRandomSampler(indices_train)
valid_sampler = SubsetRandomSampler(indices_test)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                           sampler=train_sampler,collate_fn=pad_collate)#, collate_fn=pad_collate


eval_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                               sampler=valid_sampler,collate_fn=pad_collate)# collate_fn=pad_collate

#Testing piece of code

# for ind, (eval_data, seq_len, label) in enumerate(eval_loader):
#
#     print(eval_data.shape)
#
#     if ind >= 0:
#         import sys
#         sys.exit(0)

#Testing piece of code ends here

# # Training

# load model
model = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force)

# initilize weight
with torch.no_grad():
    for child in list(model.children()):
        print(child)
        for param in list(child.parameters()):
              if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
#                     nn.init.uniform_(param, a=-0.05, b=0.05)

#check whether decoder gru weights are fixed
if fix_weight:
    print(model.decoder.gru.requires_grad)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

criterion_seq = nn.L1Loss(reduction='none')

file_output = open(root+'output/%sen%d_hid%d.txt'% (network, en_num_layers, hidden_size), 'w' )

training(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq,  file_output,
             root, network, en_num_layers, hidden_size, num_class=60,
             )


file_output.close()
