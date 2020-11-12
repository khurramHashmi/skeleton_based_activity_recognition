import os
import pickle
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class pytorch_dataloader(Dataset):

    def __init__(self, batch_size, train_path=None, test_path=None, is_train=True, split_=False, seg=30):

        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.is_train = is_train
        self.split_ = split_
        self.seg = seg
        self.read_data()

    def __len__(self):
        if self.is_train:
            return len(self.input_features)
        return len(self.test_features)

    def __getitem__(self, item):

        if self.is_train:
            return self.train_next_batch(item)
        return self.test_next_batch(item)

    def train_next_batch(self, sample_num):

        xx = []  # training batch of depth features
        yy = []  # training batch of RGB features
        zz = []  # training batch of labels
        # for sample_num in random.sample(range(self.sample_train_num), _batch_size):

        cur_label = np.zeros(60)
        cur_label[self.train_data_label[sample_num]] = 1
        cur_label = torch.tensor(cur_label, dtype=torch.long)
        zz.append(cur_label)

        class_idx = self.train_data_label[sample_num]
        indices = self.train_indices_dict[str(class_idx)].copy()

        if self.split_:
            f = h5py.File(os.path.join(self.train_path, self.input_features[random.choice(indices)]), 'r')
            first_ex = np.array(f['x'][:])
            f = h5py.File(os.path.join(self.train_path, self.input_features[random.choice(indices)]), 'r')
            second_example = np.array(f['x'][:])
        else:
            first_ex = self.input_features[random.choice(indices)]['input']
            first_ex = np.array(first_ex)
            second_example = self.input_features[random.choice(indices)]['input']
            second_example = np.array(second_example)

        xx.append(first_ex)
        yy.append(second_example)

        xx, _ = self.normalize(xx, zz)
        yy, _ = self.normalize(yy, zz)
        xx = xx.view(1, -1)
        yy = yy.view(1, -1)

        if len(xx) < 2:
            return torch.tensor(yy[0], dtype=torch.float), torch.tensor(xx[0], dtype=torch.float), zz[0]

        return yy, xx, zz
        # return torch.tensor(yy, dtype=torch.float), torch.tensor(xx, dtype=torch.float), zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, sample_num):
        xx = []  # testing batch of depth features
        yy = []  # testing batch of RGB features
        zz = []  # testing batch of labels

        #for sample_num in random.sample(range(self.sample_test_num), _batch_size):
        cur_label = np.zeros(60)
        cur_label[self.test_data_label[sample_num]] = 1
        cur_label = torch.tensor(cur_label, dtype=torch.long)
        zz.append(cur_label)
        class_idx = self.test_data_label[sample_num]
        indices = self.test_indices_dict[str(class_idx)].copy()

        if self.split_:
            first_ex= np.array(self.test_features[random.choice(indices)])['input']
            zero_pos = np.where(~first_ex.any(axis=1))[0]
            if len(zero_pos) > 0:
                zero_pos = zero_pos[0]
            else:
                zero_pos = len(first_ex)
            first_ex = first_ex[:zero_pos]
            xx.append(first_ex[:first_ex.shape[0]//2, :])
            yy.append(first_ex[first_ex.shape[0]//2:, :])
        else:
            first_ex = self.test_features[random.choice(indices)]['input']
            first_ex = np.array(first_ex)
            second_ex = self.test_features[random.choice(indices)]['input']
            second_ex = np.array(second_ex)
            xx.append(first_ex)
            yy.append(second_ex)

        xx, _ = self.normalize(xx, zz)
        yy, _ = self.normalize(yy, zz)
        xx = xx.view(1, -1)
        yy = yy.view(1, -1)

        if len(xx) < 2:
            return torch.tensor(yy[0], dtype=torch.float), torch.tensor(xx[0], dtype=torch.float), zz[0]

        return yy, xx, zz
        # return torch.tensor(yy, dtype=torch.float), torch.tensor(xx, dtype=torch.float), zz

    def read_data(self):

        if self.is_train:

            if self.split_:
                self.input_features = os.listdir(self.train_path)

            else:
                self.input_features = pickle.load(open(self.train_path, "rb"))
                self.train_data_label = []
                for j in self.input_features:
                    self.train_data_label.append(int(j['label']))

            self.train_indices_dict = {}
            for i in range(60):
                idx = []
                for j in range(len(self.train_data_label)):
                    if self.train_data_label[j] == i:
                        idx.append(j)
                self.train_indices_dict[str(i)] = idx.copy()
            self.sample_train_num = len(self.input_features)

        else:

            if self.split_:

                f = h5py.File(self.test_path, 'r')
                self.test_features = f['test_x'][:]
                self.test_data_label = list(np.argmax(f['test_y'][:], -1))
                del f

            else:
                self.test_features = pickle.load(open(self.test_path, "rb"))
                self.test_data_label = []
                for j in self.test_features:
                    self.test_data_label.append(int(j['label']))
                    # self.test_data_label.append(int(j.split('_')[-1].split('.')[0]))

            self.sample_test_num = len(self.test_features)
            self.test_indices_dict = {}
            for k in range(60):
                idx = []
                for l in range(len(self.test_data_label)):
                    if self.test_data_label[l] == k:
                        idx.append(l)
                self.test_indices_dict[str(k)] = idx.copy()

    def turn_two_to_one(self, seq):
        new_seq = list()
        for idx, ske in enumerate(seq):
            if (ske[0:75] == np.zeros((1, 75))).all():
                new_seq.append(ske[75:])
            elif (ske[75:] == np.zeros((1, 75))).all():
                new_seq.append(ske[0:75])
            else:
                new_seq.append(ske[0:75])
                new_seq.append(ske[75:])
        return np.array(new_seq)

    def Tolist_fix(self, joints, train=1):
        seqs = []
        # print(len(joints))
        for idx, seq in enumerate(joints):
            zero_row = []
            if seq.shape[1] > 75:
                seq = seq[:, :75]
            for i in range(len(seq)):

                if (seq[i, :] == np.zeros((1, 75))).all():
                    zero_row.append(i)

            seq = np.delete(seq, zero_row, axis=0)

            # seq = self.turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        return seqs

    def _rot(self, rot):
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = rot.new(rot.size()[:2] + (1,)).zero_()
        ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
        rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=2)

        ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=2)

        rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=2)

        rot = rz.matmul(ry).matmul(rx)
        return rot


    def sub_seq(self, seqs, seq, train=1):

        group = self.seg

        if seq.shape[0] < self.seg:

            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs

    def _transform(self, x, theta):
        x = x.contiguous().view(x.size()[:2] + (-1, 3))
        rot = x.new(x.size()[0], 3).uniform_(-theta, theta)
        rot = rot.repeat(1, x.size()[1])
        rot = rot.contiguous().view((-1, x.size()[1], 3))
        rot = self._rot(rot)
        x = torch.transpose(x, 2, 3)
        x = torch.matmul(rot, x)
        x = torch.transpose(x, 2, 3)
        x = x.contiguous().view(x.size()[:2] + (-1,))
        return x

    def normalize(self, first_ex, y):
        theta = 0.3
        x = self.Tolist_fix(first_ex, train=1)
        # lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
        # idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        # y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        # x = self._transform(x, theta)
        return x, y


class custom_data_loader:

    def __init__(self, train_path, test_path=None, batch_size=30, split_=False, seg=50):

        self.train_path = train_path
        self.test_path = test_path
        self.features = None
        self.data_x = [] # training and testing RGB + depth feature
        self.data_label = [] # training and testing depth label
        self.train_data_x = [] # training depth feature
        self.train_data_y = [] # training RGB feature
        self.train_data_label = [] # training label
        self.test_data_x = [] # testing depth feature
        self.test_data_y = [] # testing RGB feature
        self.test_data_label = [] # testing label
        self.train_data_xy = [] # training RGB + depth feature
        self.test_data_xy = [] # testing RGB + depth feature
        self.batch_size = batch_size
        self.seg = seg
        self.split_ = split_

    def read_train(self):

        if self.split_:

            f = h5py.File(self.test_path, 'r')
            self.input_features = os.listdir(self.train_path)
            self.test_features = f['test_x'][:]
            self.test_data_label = list(np.argmax(f['test_y'][:], -1))
            del f

        else:
            # f = h5py.File(self.train_path, 'r')
            # self.input_features = np.concatenate((f['x'][:], f['valid_x'][:]))
            # print('Total input examples ', self.input_features.shape)
            # self.train_data_label = list(np.concatenate((np.argmax(f['y'][:], -1), np.argmax(f['valid_y'][:], -1))))
            self.input_features = pickle.load(open(self.train_path, "rb"))
            self.test_features = pickle.load(open(self.test_path, "rb"))


        # assert len(self.test_data_label) == self.test_features.shape[0], print('test data and shape not same')

        # got the sample number
        self.sample_train_num = len(self.input_features)
        self.sample_test_num = len(self.test_features)

        # create dict to store indices

        # if self.split_: # use this if spilt is on
        self.train_data_label = []
        for j in self.input_features:
            self.train_data_label.append(int(j['label']))
            # self.train_data_label.append(int(j.split('_')[-1].split('.')[0]))
        #
        self.test_data_label = []
        for j in self.test_features:
            self.test_data_label.append(int(j['label']))
            # self.test_data_label.append(int(j.split('_')[-1].split('.')[0]))
        # print(self.train_data_label)

        self.train_indices_dict = {}
        for i in range(60):
            idx = []
            for j in range(len(self.train_data_label)):
                if self.train_data_label[j] == i:
                    idx.append(j)
            self.train_indices_dict[str(i)] = idx.copy()

        self.test_indices_dict = {}
        for k in range(60):
            idx = []
            for l in range(len(self.test_data_label)):
                if self.test_data_label[l] == k:
                    idx.append(l)
            self.test_indices_dict[str(k)] = idx.copy()

    def turn_two_to_one(self, seq):
        new_seq = list()
        for idx, ske in enumerate(seq):
            if (ske[0:75] == np.zeros((1, 75))).all():
                new_seq.append(ske[75:])
            elif (ske[75:] == np.zeros((1, 75))).all():
                new_seq.append(ske[0:75])
            else:
                new_seq.append(ske[0:75])
                new_seq.append(ske[75:])
        return np.array(new_seq)

    def Tolist_fix(self, joints, train=1):
        seqs = []
        # print(len(joints))
        for idx, seq in enumerate(joints):
            zero_row = []
            if seq.shape[1] > 75:
                seq = seq[:, :75]
            for i in range(len(seq)):

                if (seq[i, :] == np.zeros((1, 75))).all():
                    zero_row.append(i)

            seq = np.delete(seq, zero_row, axis=0)

            # seq = self.turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        return seqs

    def _rot(self, rot):
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = rot.new(rot.size()[:2] + (1,)).zero_()
        ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
        rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=2)

        ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=2)

        rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=2)

        rot = rz.matmul(ry).matmul(rx)
        return rot


    def sub_seq(self, seqs, seq, train=1):

        group = self.seg

        if seq.shape[0] < self.seg:

            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs

    def _transform(self, x, theta):
        x = x.contiguous().view(x.size()[:2] + (-1, 3))
        rot = x.new(x.size()[0], 3).uniform_(-theta, theta)
        rot = rot.repeat(1, x.size()[1])
        rot = rot.contiguous().view((-1, x.size()[1], 3))
        rot = self._rot(rot)
        x = torch.transpose(x, 2, 3)
        x = torch.matmul(rot, x)
        x = torch.transpose(x, 2, 3)
        x = x.contiguous().view(x.size()[:2] + (-1,))
        return x

    def normalize(self, first_ex, y):
        theta = 0.3
        x = self.Tolist_fix(first_ex, train=1)
        # lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
        # idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        # y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        # x = self._transform(x, theta)
        return x, y

    def train_next_batch(self, _batch_size):


        xx = []  # training batch of depth features
        yy = []  # training batch of RGB features
        zz = []  # training batch of labels
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):

            cur_label = np.zeros(60)
            cur_label[self.train_data_label[sample_num]] = 1
            zz.append(cur_label)

            class_idx = self.train_data_label[sample_num]
            indices = self.train_indices_dict[str(class_idx)].copy()

            if self.split_:
                f = h5py.File(os.path.join(self.train_path, self.input_features[random.choice(indices)]), 'r')
                first_ex = np.array(f['x'][:])
                f = h5py.File(os.path.join(self.train_path, self.input_features[random.choice(indices)]), 'r')
                second_example = np.array(f['x'][:])
            else:
                first_ex = self.input_features[random.choice(indices)]['input']
                first_ex = np.array(first_ex)
                second_example = self.input_features[random.choice(indices)]['input']
                second_example = np.array(second_example)

            xx.append(first_ex)
            yy.append(second_example)

        xx, _ = self.normalize(xx, zz)
        yy, _ = self.normalize(yy, zz)
        xx = xx.view(_batch_size, -1)
        yy = yy.view(_batch_size, -1)

        return yy, xx, zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, _batch_size):
        xx = []  # testing batch of depth features
        yy = []  # testing batch of RGB features
        zz = []  # testing batch of labels

        for sample_num in random.sample(range(self.sample_test_num), _batch_size):

            cur_label = np.zeros(60)
            cur_label[self.test_data_label[sample_num]] = 1
            zz.append(cur_label)
            class_idx = self.test_data_label[sample_num]
            indices = self.test_indices_dict[str(class_idx)].copy()



            if self.split_:
                first_ex= np.array(self.test_features[random.choice(indices)])['input']
                zero_pos = np.where(~first_ex.any(axis=1))[0]
                if len(zero_pos) > 0:
                    zero_pos = zero_pos[0]
                else:
                    zero_pos = len(first_ex)
                first_ex = first_ex[:zero_pos]
                xx.append(first_ex[:first_ex.shape[0]//2, :])
                yy.append(first_ex[first_ex.shape[0]//2:, :])
            else:
                first_ex = self.test_features[random.choice(indices)]['input']
                first_ex = np.array(first_ex)
                second_ex = self.test_features[random.choice(indices)]['input']
                second_ex = np.array(second_ex)
                xx.append(first_ex)
                yy.append(second_ex)

        xx, _ = self.normalize(xx, zz)
        yy, _ = self.normalize(yy, zz)
        xx = xx.view(_batch_size, -1)
        yy = yy.view(_batch_size, -1)

        return xx, yy, zz

