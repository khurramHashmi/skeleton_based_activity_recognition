import os
import wandb
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import custom_dataloaders
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader
from online_triplet_loss.losses import *
from mpl_toolkits.mplot3d import Axes3D
from visual_skeleton_3d import Draw3DSkeleton

os.environ["WANDB_MODE"] = "dryrun"
os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"

class activation_2(nn.Module):

    def __init__(self, alpha=0.2):
        super(activation_2, self).__init__()

        self.alpha = alpha
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x) - self.alpha * self.relu(x)

# class mini_batch(nn.Module):
#
#     def __init__(self, des2_l3):
#         super(mini_batch, self).__init__()
#         self.des2_l3 = des2_l3
#
#     def forward(self, input, num_kernels=5, kernel_dim=3):
#         x = self.des2_l3(input)
#         activation = torch.reshape(x, (-1, num_kernels, kernel_dim))
#         diffs = torch.unsqueeze(activation, 3) - torch.unsqueeze(torch.transpose(activation, [1, 2, 0]), 0)
#         abs_diffs = torch.sum(torch.abs(diffs), dim=2)
#         minibatch_features = torch.sum(torch.exp(-abs_diffs), dim=2)
#
#         return torch.cat([input, minibatch_features], dim=1)
# Triplet Semihard pytorch loss xxx

class LSTM_Gen(nn.Module):

    def __init__(self, input_size,hidden_size, num_layers=2):
        super(LSTM_Gen, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size, 128, num_layers,batch_first=True, dropout=0.5)
        self.lstm_2 = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=0.5)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), 128).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), 128).to(device)
        x, _ = self.lstm_1(x, (h0, c0))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        x, _ = self.lstm_2(x, (h0, c0))
        return x

class LSTM_Encoders(nn.Module):

    def __init__(self, input_size,hidden_size, num_layers=2):
        super(LSTM_Encoders, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers * 2 , x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        x, _ = self.lstm_1(x, (h0, c0))
        return x


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss



class cyclegan(nn.Module):

    def __init__(self, num_classes, batch_size, learning_rate, device, seg=30):

        super(cyclegan, self).__init__()

        self.batch_size = batch_size
        self.lr = learning_rate
        self.x_dim = seg * 75
        self.y_dim = seg * 75
        self.seg = seg

        self.num_classes = num_classes
        self.hx_dim = 800 #1024 # Encoder 1 hidden dimension
        self.hy_dim = 800 #1024 # Encoder 2 hidden dimension
        self.space_dim_1 = 128 #256 # First Subspace dimension
        self.space_dim_2 = 128 #256 # Second Subspace dimension

        self.noise_dim = 300 #256 # Noise dimension same as the sub space dimensions
        self.epsilon = 1e-6

        self.hd1_dim = 400 #256 # First Discriminator hidden dimension
        self.hd2_dim = 400 #256 # Second Discriminator hidden dimension

        # self.hg1_dim = 512
        # self.hg2_dim = 512

        self.h_g_dim = 500 #512 # Only one Dim for both of the Generators

        self.g_updates = 5
        self.c_rf = 0.99  # weights between real and fake
        self.cg_rf = 0.9  # the weights of the Cg() model (looks like =1 is more reasonable)
        self.hcg_dim = 100  # graph classifier Cg() hidden layer dimension for VRDCN
        self.lamda_g_smi = 0.1  # similarity weights
        self.lbd_t = 0.001  # triplet loss weight

        self.leaky_relu = nn.LeakyReLU(0.25)

        self.activation_25 = activation_2(0.25)
        self.activation_2 = activation_2()

        self.lstm = LSTM_Gen(75,75,2)
        self.lstm_encoder = LSTM_Encoders(75,self.space_dim_1,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.softmax_criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.L1Loss()
        self.gans_criterion = nn.MSELoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()

        self.triplet_criterion = HardTripletLoss()

        # self.sigmoid_criterion = F.binary_cross_entropy_with_logits()
        # self.triplet_criterion = batch_hard_triplet_loss()
        self.create_network()
        self.device = device

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        # print('Initializing layer')
        # torch.nn.init.xavier_uniform(m.weight)

    def sample_Noise(self, m, n):
        return np.random.uniform(0., 1., size=[m, n])

    def create_encoder_1(self):

        # self.encoder_l1= nn.Linear(self.x_dim, self.hx_dim)
        # self.encoder_l2 = nn.Linear(self.hx_dim, self.space_dim_1)
        self.encoder1 = nn.Sequential(*[self.lstm_encoder]).cuda()

        return None

    def create_encoder_2(self):
        # self.encoder2_l1 = nn.Linear(self.y_dim, self.hy_dim)
        # self.encoder2_l2 = nn.Linear(self.hy_dim, self.space_dim_2)
        # self.encoder2 = nn.Sequential(*[self.encoder2_l1, self.activation_25, self.encoder2_l2, self.activation_25]).cuda()

        self.encoder2 = nn.Sequential(*[self.lstm_encoder]).cuda()

        return None

    def encoder_forward(self, encoder_type, x):

        x = x.view(self.batch_size, self.seg, 75)

        if encoder_type == 1:
            # x = self.sigmoid(x)
            # x = self.activation(self.encoder_l1(x))
            # return self.activation(self.encoder_l2(x))
            x = self.encoder1(x)
        else:
            # x = self.activation(self.encoder2_l1(x))
            # return self.activation(self.encoder2_l2(x))
            x = self.encoder2(x)

        x = x[:, -1, :]

        return x

    def create_classifier1(self):

        self.classifier1_l1 = nn.Linear(self.space_dim_1*2, self.num_classes)
        self.classifier1 = nn.Sequential(*[self.classifier1_l1]).cuda()
        return [self.classifier1_l1]

    def classifier1_forward(self, x):

        Classifier_logit = self.classifier1(x)  # self.classifier1_l1(x)
        Classifier_prob = self.softmax(Classifier_logit) #Classifier_prob = self.sigmoid(Classifier_logit)
        return Classifier_logit, Classifier_prob

    def create_classifier2(self):

        self.classifier2_l1 = nn.Linear(self.space_dim_2*2, self.num_classes)
        self.classifier2 = nn.Sequential(*[self.classifier2_l1]).cuda()
        return [self.classifier2_l1]

    def classifier2_forward(self, x):

        Classifier_logit = self.classifier2(x)  # self.classifier2_l1(x)
        Classifier_prob = self.softmax(Classifier_logit) #Classifier_prob = self.sigmoid(Classifier_logit)
        return Classifier_logit, Classifier_prob

    def create_descr1(self):

        self.des1_l1 = nn.Linear(self.y_dim + self.num_classes, self.hd1_dim)
        self.des1_l2 = nn.Linear(self.hd1_dim, self.hd1_dim)
        self.des1_l3 = nn.Linear(self.hd1_dim, 15)
        self.des1_l4 = nn.Linear(405, 1)
        self.desc1_1 = nn.Sequential(*[self.des1_l1, self.activation_2, self.des1_l2, self.activation_2, self.des1_l3]).cuda()
        self.desc1_2 = nn.Sequential(*[self.des1_l4]).cuda()
        return [self.des1_l1, self.des1_l2, self.des1_l3, self.des1_l4]

    def minibatch1(self, previous_input, input, num_kernels=5, kernel_dim=3):

        activation = torch.reshape(input, (-1, num_kernels, kernel_dim))
        a = torch.unsqueeze(activation, 3)
        b = torch.unsqueeze(torch.transpose(activation, dim0=1, dim1=2), 0)
        b = b.permute(0, 3, 2, 1)
        diffs = a-b # [1, 2, 0]), 0)
        abs_diffs = torch.sum(torch.abs(diffs), dim=2)
        minibatch_features = torch.sum(torch.exp(-abs_diffs), dim=2)
        return torch.cat([previous_input, minibatch_features], dim=1)

    def des1_forward(self, y, z):

        inputs = torch.cat([y, z], dim=1)
        x1 = self.desc1_1(inputs) # get output till layer 3 batch_size x 15
        temp_model = nn.Sequential(*list(self.desc1_1.children())[:-1]).cuda()
        x = temp_model(inputs)
        x = self.activation_2.forward(x)
        x = self.minibatch1(x, x1) # apply minibatch
        x = self.desc1_2(x) # get output from final layer 4
        D_logit = x
        x = self.sigmoid(x)

        return x, D_logit

    def create_descr2(self):
        self.des2_l1 = nn.Linear(self.x_dim + self.num_classes, self.hd2_dim)
        self.des2_l2 = nn.Linear(self.hd2_dim, self.hd2_dim)
        self.des2_l3 = nn.Linear(self.hd2_dim, 15)
        # self.des2_l4 = nn.Linear(15, 405)
        self.des2_l4 = nn.Linear(405, 1)
        # self.minibatch2 = mini_batch(self.des2_l3)
        self.desc2_1 = nn.Sequential(*[self.des2_l1, self.activation_2, self.des2_l2, self.activation_2, self.des2_l3]).cuda()
        self.desc2_2 = nn.Sequential(*[self.des2_l4]).cuda()
        return [self.des2_l1, self.des2_l2, self.des2_l3, self.des2_l4]


    def des2_forward(self, y, z):

        inputs = torch.cat([y, z], dim=1)
        x1 = self.desc2_1(inputs)
        temp_model = nn.Sequential(*list(self.desc2_1.children())[:-1]).cuda()
        x = temp_model(inputs)
        x = self.activation_2.forward(x)
        x = self.minibatch1(x, x1)
        x = self.desc2_2(x)
        D_logit = x
        x = self.sigmoid(x)

        return x, D_logit


    def create_gen1(self):



        # self.gen1_l1 = nn.Linear(self.x_dim,self.h_g_dim, bias=False)
        # self.gen1_l2 = nn.Linear(self.h_g_dim,self.h_g_dim, bias=False)
        # self.gen1_l3 = nn.Linear(self.h_g_dim, self.h_g_dim)
        # self.gen1_l4 = nn.Linear(self.h_g_dim, self.y_dim)
        # self.batch_norm1_g1 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        # self.batch_norm2_g1 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        # self.gen_ones = torch.ones(self.h_g_dim)
        # self.gen_zeros = torch.zeros(self.h_g_dim)

        self.gen1 = nn.Sequential(*[self.lstm]).cuda()

        # self.gen1 = nn.Sequential(*[self.gen1_l1, self.batch_norm1_g1, self.activation_2, self.gen1_l2, self.batch_norm2_g1, self.gen1_l3, self.activation_2, self.gen1_l4]).cuda()

        return None


    def gen1_forward(self, x, noise):
        # noise = noise * 60
        # inputs = torch.cat([x, noise], dim=1)
        # print(type(inputs))
        # inputs = torch.tensor(inputs, dtype=torch.double)
        # x = self.gen1_l1(inputs)
        # x = self.batch_norm(x)
        # x = self.gen_ones * x + self.gen_zeros
        # x = self.activation_2(x,0.2)
        # x = self.gen1_l2(x)
        # x = self.batch_norm2(x)
        # x = self.activation_2(self.gen1_l3(x), 0.2)
        # x = self.gen1_l4(x)
        x = x.view(self.batch_size, self.seg, 75)

        x = self.gen1(x)
        x = x.reshape(self.batch_size, -1)

        return x

    def create_gen2(self):
        # self.gen2_l1 = nn.Linear(self.y_dim ,self.h_g_dim, bias=False)
        # self.gen2_l2 = nn.Linear(self.h_g_dim,self.h_g_dim, bias=False)
        # self.gen2_l3 = nn.Linear(self.h_g_dim, self.h_g_dim)
        # self.gen2_l4 = nn.Linear(self.h_g_dim, self.x_dim)
        # self.batch_norm1_g2 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        # self.batch_norm2_g2 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        # self.gen_ones = torch.ones(self.h_g_dim)
        # self.gen_zeros = torch.zeros(self.h_g_dim)
        # self.gen2 = nn.Sequential(*[self.gen2_l1, self.batch_norm1_g2, self.activation_2, self.gen2_l2, self.batch_norm2_g2, self.gen2_l3, self.activation_2, self.gen2_l4]).cuda()

        self.gen2 = nn.Sequential(*[self.lstm]).cuda()

        return None

    def gen2_forward(self, x, noise):
        # noise = noise * 60
        # inputs = torch.cat([x, noise], dim=1)
        # x = self.gen2_l1(inputs)
        # x = self.batch_norm1(x)
        # x = self.gen_ones * x + self.gen_zeros
        # x = self.activation_2(x,0.2)
        # x = self.gen2_l2(x)
        # x = self.batch_norm2(x)
        # x = self.activation_2(self.gen2_l3(x), 0.2)
        # x = self.gen2_l4(x)
        # return x

        x = x.view(self.batch_size, self.seg, 75)
        x = self.gen2(x)
        x = x.reshape(self.batch_size, -1)

        return x

    def create_vrdn(self):

        self.vrdn_l1 = nn.Linear(self.num_classes * self.num_classes,self.hcg_dim)
        self.vrdn_l2 = nn.Linear(self.hcg_dim, self.num_classes)
        self.vrdn = nn.Sequential(*[self.vrdn_l1, self.activation_25, self.vrdn_l2]).cuda()

        return [self.vrdn_l1,  self.vrdn_l2]

    def vrdn_forward(self, x, y):

        # make correlation matrix
        C_prob_1 = torch.unsqueeze(x, -1)
        C_prob_2 = torch.unsqueeze(y, 1)
        W_feature = torch.matmul(C_prob_1, C_prob_2)
        C_hw = torch.reshape(W_feature, [-1, self.num_classes * self.num_classes])
        # fordward pass
        Classifier_g_logit = self.vrdn(C_hw)
        # x = self.activation_2(self.vrdn_l1(C_hw), 0.25)
        # Classifier_g_logit = self.vrdn_l2(x)
        Classifier_g_prob = self.softmax(Classifier_g_logit)

        return Classifier_g_logit, Classifier_g_prob


    def create_network(self):

        enc1_vars = self.create_encoder_1()
        enc2_vars = self.create_encoder_2()
        vrdn_vars = self.create_vrdn()
        gen1_vars = self.create_gen1()
        gen2_vars = self.create_gen2()
        des1_vars = self.create_descr1()
        des2_vars = self.create_descr2()
        classifier1_vars = self.create_classifier1()
        classifier2_vars = self.create_classifier2()

        self.encoder1.apply(self.xavier_init)
        self.encoder2.apply(self.xavier_init)
        self.classifier1.apply(self.xavier_init)
        self.classifier2.apply(self.xavier_init)
        self.vrdn.apply(self.xavier_init)
        self.gen1.apply(self.xavier_init)
        self.gen2.apply(self.xavier_init)
        self.desc1_1.apply(self.xavier_init)
        self.desc1_2.apply(self.xavier_init)
        self.desc2_1.apply(self.xavier_init)
        self.desc2_2.apply(self.xavier_init)

    def calculate_accuracy(self, class_total, class_correct,  mode):
        acc_sum = 0
        for i in range(len(classes)):
            if class_total[i] == 0:
                class_accuracy = 0
            else:
                class_accuracy = 100 * class_correct[i] / class_total[i]
            acc_sum += class_accuracy
            print(mode + ' %d Accuracy of %5s : %2d %% and total count : %2d ' % (i, classes[i], class_accuracy, class_total[i]))
        print('=' * 89)
        print('Mean Average Accuracy of Camera View : %2f %%' % (acc_sum / 60))
        print('=' * 89)

        return acc_sum / len(classes)

    def class_accuracy(self, pred, labels, class_total, class_correct):

        c = (pred == labels).squeeze()

        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] = class_total[label] + 1

        return class_total, class_correct


    def forward(self, subject_1, subject_2, labels, noise, class_total, class_correct, eval_mode=False):

        generate_fake_z2 = self.gen1_forward(subject_1, noise)  # generate S1 -> S2
        generate_fake_z1 = self.gen2_forward(subject_2, noise)  # generate S2 -> S1

        C_1_real_logit, C_1_real_prob = self.classifier1_forward(self.encoder_forward(1, subject_1))
        C_1_fake_logit, C_1_fake_prob = self.classifier1_forward(self.encoder_forward(1, generate_fake_z1))

        C_2_real_logit, C_2_real_prob = self.classifier2_forward(self.encoder_forward(2, subject_2))
        C_2_fake_logit, C_2_fake_prob = self.classifier2_forward(self.encoder_forward(2, generate_fake_z2))

        C_g_logit_real, C_g_prob_real = self.vrdn_forward(C_1_real_prob, C_2_real_prob)
        C_g_logit_rf, C_g_prob_rf = self.vrdn_forward(C_1_real_prob, C_2_fake_prob)
        C_g_logit_fr, C_g_prob_fr = self.vrdn_forward(C_1_fake_prob, C_2_real_prob)

        # calculate classiciation loss for separate Classififers
        # Will Try Cross Entropy Since VRDCN is already on Cross Entropy
        # C_1_loss = self.softmax_criterion(C_1_real_logit, C_1_fake_logit, labels)
        # C_2_loss = self.softmax_criterion(C_2_real_logit, C_2_fake_logit, labels)
        C_1_loss = self.compute_classifier_loss(C_1_real_logit, C_1_fake_logit, labels)

        C_2_loss = self.compute_classifier_loss(C_2_real_logit, C_2_fake_logit, labels)

        # calculate classiciation loss for Correlation Classififer
        C_g_loss_real = self.softmax_criterion(C_g_logit_real, torch.argmax(labels, 1))
        C_g_loss_rf = self.softmax_criterion(C_g_logit_rf, torch.argmax(labels, 1))
        C_g_loss_fr = self.softmax_criterion(C_g_logit_fr, torch.argmax(labels, 1))

        C_g_loss_sum = self.cg_rf * C_g_loss_real + (1.0 - self.cg_rf) / 2 * (C_g_loss_rf + C_g_loss_fr)

        # check this
        # triplet_loss_1 = batch_hard_triplet_loss(torch.argmax(labels, 1),
        #                                          F.normalize(self.encoder_forward(1, subject_1), p=2), margin=1.0, device='cuda')

        triplet_loss_1 = self.triplet_criterion.forward(self.encoder_forward(1, subject_1), torch.argmax(labels, 1))

        triplet_loss_2 = self.triplet_criterion.forward(self.encoder_forward(2, subject_2), torch.argmax(labels, 1))

        # triplet_loss_2 = batch_hard_triplet_loss(torch.argmax(labels, 1), F.normalize(self.encoder_forward(2, subject_2), p=2),
        # margin=1.0, device='cuda')

        E_1_loss = self.lbd_t * triplet_loss_1 + C_1_loss
        E_2_loss = self.lbd_t * triplet_loss_2 + C_2_loss

        # discriminator 1 loss
        D1_prob_real, D1_logit_real = self.des1_forward(subject_2, labels)
        D1_prob_fake, D1_logit_fake = self.des1_forward(generate_fake_z2, labels)

        D1_loss_real = self.binary_criterion(D1_logit_real, torch.ones_like(D1_logit_real))
        D1_loss_fake = self.binary_criterion(D1_logit_fake, torch.zeros_like(D1_logit_fake))

        D1_loss = D1_loss_real + D1_loss_fake

        # discriminator 2 loss
        D2_prob_real, D2_logit_real = self.des2_forward(subject_1, labels)
        D2_prob_fake, D2_logit_fake = self.des2_forward(generate_fake_z1, labels)

        D2_loss_real = self.binary_criterion(D2_logit_real, torch.ones_like(D2_logit_real))
        D2_loss_fake = self.binary_criterion(D2_logit_fake, torch.zeros_like(D2_logit_fake))

        D2_loss = D2_loss_real + D2_loss_fake

        # call these ops in testing part on train data
        D1_real_ave = torch.mean(D1_prob_real)
        D1_fake_ave = torch.mean(D1_prob_fake)
        D2_real_ave = torch.mean(D2_prob_real)
        D2_fake_ave = torch.mean(D2_prob_fake)

        # generator 1 and generator 2 loss
        # G1_loss_dis = self.binary_criterion(D1_logit_fake, torch.ones_like(D1_logit_fake))
        # G2_loss_dis = self.binary_criterion(D2_logit_fake, torch.ones_like(D2_logit_fake))
        G1_loss_dis = self.gans_criterion(D1_logit_fake, torch.ones_like(D1_logit_fake))
        G2_loss_dis = self.gans_criterion(D2_logit_fake, torch.ones_like(D2_logit_fake))
        G1_loss_fea = self.reconstruction_criterion(generate_fake_z2,
                                                  subject_2)  # self.reconstruction_criterion(generate_fake_z2, subject_2) # generated feature similiarity
        G2_loss_fea = self.reconstruction_criterion(generate_fake_z1,
                                                  subject_1)  # self.reconstruction_criterion(generate_fake_z1, subject_1) # generated feature similiarity
        G1_loss = G1_loss_dis + self.lamda_g_smi * G1_loss_fea
        G2_loss = G2_loss_dis + self.lamda_g_smi * G2_loss_fea

        # if eval_mode: # need to calculate Test losses as well
            # compute accuracies for classifiers
        acc_c1_real = torch.mean(torch.eq(torch.argmax(C_1_real_logit, 1), torch.argmax(labels, 1)).float())
        acc_c1_fake = torch.mean(torch.eq(torch.argmax(C_1_fake_logit, 1), torch.argmax(labels, 1)).float())
        acc_c2_real = torch.mean(torch.eq(torch.argmax(C_2_real_logit, 1), torch.argmax(labels, 1)).float())
        acc_c2_fake = torch.mean(torch.eq(torch.argmax(C_2_fake_logit, 1), torch.argmax(labels, 1)).float())
        # Evaluate on testing stage in 3 settings: (1) real Z1 real Z2 (2) real Z1 fake Z2 (3) fake Z1 real Z2
        C_r1r2_logit, C_r1r2_prob = self.vrdn_forward(C_1_real_prob, C_2_real_prob)  # real z1 + real z2
        acc_te_r1r2 = torch.mean(torch.eq(torch.argmax(C_r1r2_prob, 1), torch.argmax(labels, 1)).float())
        # calculate per class accuracy
        y_true = torch.argmax(labels, 1).detach().cpu().numpy()
        y_pred = torch.argmax(C_r1r2_prob, 1).detach().cpu().numpy()
        class_total, class_correct = self.class_accuracy(y_pred, y_true, class_total, class_correct)

        # y_pred = [classes[i] for i in torch.argmax(C_r1r2_prob, 1).detach().cpu().numpy()]
        # y_true = [classes[i] for i in torch.argmax(labels, 1).detach().cpu().numpy()]
        # draw_confusion_matrix(y_true, y_pred, 'test', np.unique(y_true))
        C_r1f2_logit, C_r1f2_prob = self.vrdn_forward(C_1_real_prob, C_2_fake_prob)  # real z1 + fake z2
        acc_te_r1f2 = torch.mean(torch.eq(torch.argmax(C_r1f2_prob, 1), torch.argmax(labels, 1)).float())

        C_f1r2_logit, C_f1r2_prob = self.vrdn_forward(C_1_fake_prob, C_2_real_prob)  # fake z1 + real z2
        acc_te_f1r2 = torch.mean(torch.eq(torch.argmax(C_f1r2_prob, 1), torch.argmax(labels, 1)).float())

            # first is rgb-d, rgb, depth => s1-s2, s1, s2
            # return acc_te_r1r2, acc_te_r1f2, acc_te_f1r2, acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake, G1_loss, G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss

        return acc_te_r1r2, acc_te_r1f2, acc_te_f1r2, acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake, G1_loss, \
               G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss, class_total, class_correct

    def compute_generator_loss(self, fake, real_subj):

        return torch.mean(torch.norm(fake-real_subj, p=None))

    def compute_classifier_loss(self, real_logits, fake_logits, labels):

        return self.c_rf * self.softmax_criterion(real_logits, torch.argmax(labels, 1)) + (1.0 - self.c_rf) * self.softmax_criterion(fake_logits, torch.argmax(labels, 1))

        # return self.c_rf * torch.mean(torch.square(real_logits - labels)) + (1.0 - self.c_rf) * \
        #        torch.mean(torch.square(fake_logits - labels))


    def train_(self, epochs, train_loader, test_loader=None, out_dir=''):

        # visual = Draw3DSkeleton(save_path='./temp')
        # visual1 = Draw3DSkeleton(save_path='./temp_y')
        # for train_x, train_y, labels in train_loader:
        #     visual.visual_skeleton_batch(train_x.numpy(), torch.argmax(labels, 1).numpy(), self.seg)
        #     visual1.visual_skeleton_batch(train_y.numpy(), torch.argmax(labels, 1).numpy(), self.seg)
        #     break
        # import sys
        # sys.exit(0)
        wandb.init(project="Cycle-Gan", reinit=True)

        best_val_acc = 0.0
        module_names = [self.encoder1, self.encoder2, self.desc1_1, self.desc1_2,
                        self.desc2_1, self.desc2_2, self.vrdn, self.gen1, self.gen2, self.classifier1, self.classifier2]
        module_str = ['encoder_1', 'encoder_2', 'desc1_1',
                      'desc1_2', 'desc2_1', 'desc2_2', 'vrdn', 'gen1', 'gen2', 'classifier_1', 'classifier_2']
        optimizer_list = []

        # # define optimizers
        for module, m_str in zip(module_names[:-2], module_str[:-2]):
            if m_str == 'gen1':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=3e-5))
            elif m_str == 'gen2':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=3e-5))
            elif m_str == 'vrdn':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=1e-3))
            else:
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=self.lr))

        for epoch in range(epochs):

            acc_te_r1r2_num, acc_te_r1f2_num, acc_te_f1r2_num, acc_c1_r_sum, acc_c1_f_sum, acc_c2_r_sum, acc_c2_f_sum = [], [], [], [], [], [], []
            G1_loss, G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss_sum, C_2_loss_sum = [], [], [], [], [], [], [], [], []

            class_correct = list(0. for i in range(60))
            class_total = list(0. for i in range(60))

            # train_x, train_y, labels = train_loader.train_next_batch(self.batch_size)
            for i, (train_x, train_y, labels) in tqdm(enumerate(train_loader)):

                for op in optimizer_list:
                    op.zero_grad()

                noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim), dtype=torch.float)
                sub_1_acc, sub_2_acc, sub_1_2_acc, acc_c1_r, acc_c1_f, acc_c2_r, \
                acc_c2_f, g1_l, g2_l, d1_l, d2_l, e1_l, e2_l, cg_l, c_1_l, c_2_l, class_total, class_correct = \
                    self.forward(train_x.to(self.device), train_y.to(self.device), labels.to(self.device),
                                 noise_sample.to(device), class_total, class_correct)

                g1_l.backward(retain_graph=True)
                g2_l.backward(retain_graph=True)
                d1_l.backward(retain_graph=True)
                d2_l.backward(retain_graph=True)
                e1_l.backward(retain_graph=True)
                e2_l.backward(retain_graph=True)
                cg_l.backward(retain_graph=True)
                # c_1_l.backward(retain_graph=True)
                # c_2_l.backward(retain_graph=True)

                acc_te_r1r2_num.append(sub_1_acc.item())
                acc_te_r1f2_num.append(sub_2_acc.item())
                acc_te_f1r2_num.append(sub_1_2_acc.item())
                acc_c1_r_sum.append(acc_c1_r.item())
                acc_c1_f_sum.append(acc_c1_f.item())
                acc_c2_r_sum.append(acc_c2_r.item())
                acc_c2_f_sum.append(acc_c2_f.item())
                G1_loss.append(g1_l.item())
                G2_loss.append(g2_l.item())
                D1_loss.append(d1_l.item())
                D2_loss.append(d2_l.item())
                E_1_loss.append(e1_l.item())
                E_2_loss.append(e2_l.item())
                C_g_loss_sum.append(cg_l.item())
                C_1_loss_sum.append(c_1_l.item())
                C_2_loss_sum.append(c_2_l.item())

                for op in optimizer_list:
                    op.step()


            print('\n\n[TRAIN] Epoch = ', epoch, '  Loss_G_1 = %.4f' % np.mean(G1_loss), ' Loss_G_2 = %.4f' % np.mean(G2_loss),
                  '  Loss_D_1 = %.4f' % np.mean(D1_loss), '  Loss_D_2 = %.4f' % np.mean(D2_loss),
                  ' Loss_E_1 = %.4f' % np.mean(E_1_loss),
                  '  Loss_E_2 = %.4f' % np.mean(E_2_loss), '  Loss_VRDCN = %.4f' % np.mean(C_g_loss_sum),
                  '  Loss_C_1 = %.4f' % np.mean(C_1_loss_sum), '  Loss_C_2 = %.4f' % np.mean(C_2_loss_sum))
            print('\n[TRAIN] Epoch = ', epoch, '  Accuracy:  Subject_1_2 = %.4f' % np.mean(acc_te_r1r2_num),
                      ' Subject_1 = %.4f' % np.mean(acc_te_r1f2_num), '  Subject_2 = %.4f' % np.mean(acc_te_f1r2_num)
                      , ' C_1_R = %.4f' % np.mean(acc_c1_r_sum), ' C_1_F = %.4f' % np.mean(acc_c1_f_sum),
                      ' C_2_R = %.4f' % np.mean(acc_c2_r_sum), ' C_2_F = %.4f' % np.mean(acc_c2_f_sum))

            log_dict = {'Generator_1': np.mean(G1_loss), 'Generator_2': np.mean(G2_loss), 'Desc_1': np.mean(D1_loss)
                , 'Desc2': np.mean(D2_loss), 'E1': np.mean(E_1_loss), "E2": np.mean(E_2_loss),
                        'VRDN': np.mean(C_g_loss_sum),
                        'C1': np.mean(C_1_loss_sum), "C2": np.mean(C_2_loss_sum)}
            wandb.log(log_dict)
            _ = self.calculate_accuracy(class_total, class_correct, '[TRAIN]')

            if epoch % 5 == 0: #and epoch != 0:

                with torch.no_grad():
                    class_correct_eval = list(0. for i in range(60))
                    class_total_eval = list(0. for i in range(60))
                    acc_te_r1r2_num, acc_te_r1f2_num, acc_te_f1r2_num, acc_c1_r_sum, acc_c1_f_sum, acc_c2_r_sum, acc_c2_f_sum= [], [], [], [], [], [], []
                    G1_loss, G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss_sum, C_2_loss_sum = [], [], [], [], [], [], [], [], []

                    for test_x, test_y, test_labels in test_loader:
                        noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim), dtype=torch.float)
                        sub_1_acc, sub_2_acc, sub_1_2_acc, acc_c1_r, acc_c1_f, acc_c2_r, \
                        acc_c2_f, g1_l, g2_l, d1_l, d2_l, e1_l, e2_l, cg_l, c_1_l, c_2_l, class_total_eval, class_correct_eval \
                            = self.forward(test_x.to(self.device), test_y.to(self.device), test_labels.to(self.device),
                                           noise_sample.to(device), class_total_eval, class_correct_eval, eval_mode=True)

                        acc_te_r1r2_num.append(sub_1_acc.item())
                        acc_te_r1f2_num.append(sub_2_acc.item())
                        acc_te_f1r2_num.append(sub_1_2_acc.item())
                        acc_c1_r_sum.append(acc_c1_r.item())
                        acc_c1_f_sum.append(acc_c1_f.item())
                        acc_c2_r_sum.append(acc_c2_r.item())
                        acc_c2_f_sum.append(acc_c2_f.item())
                        G1_loss.append(g1_l.item())
                        G2_loss.append(g2_l.item())
                        D1_loss.append(d1_l.item())
                        D2_loss.append(d2_l.item())
                        E_1_loss.append(e1_l.item())
                        E_2_loss.append(e2_l.item())
                        C_g_loss_sum.append(cg_l.item())
                        C_1_loss_sum.append(c_1_l.item())
                        C_2_loss_sum.append(c_2_l.item())

                    print('\n [TEST] Epoch = ', epoch, '  Accuracy:  Subject_1_2 = %.4f' % np.mean(acc_te_r1r2_num),
                          ' Subject_1 = %.4f' % np.mean(acc_te_r1f2_num), '  Subject_2 = %.4f' % np.mean(acc_te_f1r2_num)
                          , ' C_1_R = %.4f' % np.mean(acc_c1_r_sum), ' C_1_F = %.4f' % np.mean(acc_c1_f_sum),
                          ' C_2_R = %.4f' % np.mean(acc_c2_r_sum), ' C_2_F = %.4f' % np.mean(acc_c2_f_sum))
                        # print('Epoch = ', epoch, '  Accuracy:  Subject_1 = %.4f' % sub_1_acc, ' Subject_2 = %.4f' % sub_2_acc,
                        #       '  Subject_1_2 = %.4f' % sub_1_2_acc)
                    print('\n [TEST] Epoch = ', epoch, '  Loss_G_1 = %.4f' % np.mean(G1_loss),
                          ' Loss_G_2 = %.4f' % np.mean(G2_loss),
                          '  Loss_D_1 = %.4f' % np.mean(D1_loss), '  Loss_D_2 = %.4f' % np.mean(D2_loss),
                          ' Loss_E_1 = %.4f' % np.mean(E_1_loss),
                          '  Loss_E_2 = %.4f' % np.mean(E_2_loss), '  Loss_VRDCN = %.4f' % np.mean(C_g_loss_sum),
                          '  Loss_C_1 = %.4f' % np.mean(C_1_loss_sum), '  Loss_C_2 = %.4f' % np.mean(C_2_loss_sum))

                    acc_dict = {'Subject_1_2': np.mean(acc_te_r1r2_num), 'Subject_1': np.mean(acc_te_r1f2_num),
                                'Subject_2':np.mean(acc_te_f1r2_num)}
                    wandb.log(acc_dict)
                    _ = self.calculate_accuracy(class_total_eval, class_correct_eval, '[TEST]')

                    if np.mean(acc_te_r1r2_num) > best_val_acc:
                        best_val_acc = np.mean(acc_te_r1r2_num)
                        best_epoch = epoch
                        for op, module, name in zip(optimizer_list, module_names, module_str):
                            torch.save({
                                'epoch': best_epoch,
                                'model_state_dict': module.state_dict(),
                                'optimizer_state_dict': op.state_dict(),
                            }, os.path.join(out_dir, name))

classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up",
           "clapping", "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe",
           "take off a shoe", "put on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
           "hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "phone call",
           "play with phone/tablet", "type on a keyboard", "point to something", "taking a selfie",
           "check time (from watch)", "rub two hands", "nod head/bow", "shake head", "wipe face", "salute",
           "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down", "headache",
           "chest pain", "back pain", "neck pain", "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing",
           "pat on back", "point finger", "hugging", "giving object", "touch pocket", "shaking hands",
           "walking towards", "walking apart"]

seg=50
max_epochs=300
batch_size = 1024
learning_rate = 1e-4
eval_batch_size = 1024
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
out_path = '/home/hashmi/Desktop/model_experiments/cg_lstm_'
if not os.path.exists(out_path):
    os.mkdir(out_path)

train_path ='/home/hashmi/Desktop/dataset/NTURGBD-60_120/ntu_60/cross_subject_data/trans_train_data.pkl'
test_path = '/home/hashmi/Desktop/dataset/NTURGBD-60_120/ntu_60/cross_subject_data/trans_test_data.pkl'
train_dataset = custom_dataloaders.pytorch_dataloader(batch_size, train_path=train_path, seg=seg, split_=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
eval_dataset = custom_dataloaders.pytorch_dataloader(eval_batch_size, test_path=test_path, is_train=False, seg=seg, split_=False)
eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=True, **kwargs)

model = cyclegan(num_classes, batch_size, learning_rate, device, seg=seg).to(device)
model.train_(max_epochs, train_loader, eval_loader, out_dir=out_path)