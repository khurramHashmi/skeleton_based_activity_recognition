import os
import torch
import numpy as np
from torch import nn
import dataset_loader
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from online_triplet_loss.losses import *

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

class TripletSemihardLoss(nn.Module):
    """
    the same with tf.triplet_semihard_loss
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self):
        super(TripletSemihardLoss, self).__init__()

    def cudafy(self, module):
        if torch.cuda.is_available():
            return module.cuda()
        else:
            return module.cpu()

    def masked_maximum(self, data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the maximum.
            Returns:
              masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
            """
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
        return masked_maximums

    def masked_minimum(self, data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the minimum.
            Returns:
              masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
            """
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
        return masked_minimums

    def pairwise_distance(self, embeddings, squared=True):
        pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                     torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                     2.0 * torch.matmul(embeddings, embeddings.t())

        error_mask = pairwise_distances_squared <= 0.0
        if squared:
            pairwise_distances = pairwise_distances_squared.clamp(min=0)
        else:
            pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

        pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

        num_data = embeddings.shape[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(self.cudafy(torch.ones([num_data])))
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
        return pairwise_distances

    def forward(self, embeddings, target, margin=1.0, squared=True):
        """
        :param features: [B * N features]
        :param target: [B]
        :param square: if the distance squared or not.
        :return:
        """
        lshape = target.shape
        assert len(lshape) == 1
        labels = target.int().unsqueeze(-1)  # [B, 1]
        pdist_matrix = self.pairwise_distance(embeddings, squared=squared)

        adjacency = labels == torch.transpose(labels, 0, 1)

        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        # Compute the mask

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
            torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                                   0.0, [batch_size, batch_size])
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # negatives_inside: largest D_an.
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.float() - torch.diag(self.cudafy(torch.ones([batch_size])))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)

        # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
        return triplet_loss



class cyclegan(nn.Module):

    def __init__(self, num_classes, batch_size, learning_rate, device, seg=30):

        super(cyclegan, self).__init__()

        self.batch_size = batch_size
        self.lr = learning_rate
        self.x_dim = 3 * 2048
        self.y_dim = 110
        self.num_classes = num_classes
        self.hx_dim = 800 #1024 # Encoder 1 hidden dimension
        self.hy_dim = 100 #1024 # Encoder 2 hidden dimension
        self.space_dim_1 = 300 #256 # First Subspace dimension
        self.space_dim_2 = 100 #256 # Second Subspace dimension

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

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.softmax_criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.L1Loss()
        self.gans_criterion = nn.MSELoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()

        self.triplet_criterion = TripletSemihardLoss()

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

        self.encoder_l1= nn.Linear(self.x_dim, self.hx_dim)
        self.encoder_l2 = nn.Linear(self.hx_dim, self.space_dim_1)
        self.encoder1 = nn.Sequential(*[self.encoder_l1, self.activation_25, self.encoder_l2, self.activation_25]).cuda()
        return [self.encoder_l1, self.encoder_l2]

    def create_encoder_2(self):
        self.encoder2_l1 = nn.Linear(self.y_dim, self.hy_dim)
        self.encoder2_l2 = nn.Linear(self.hy_dim, self.space_dim_2)
        self.encoder2 = nn.Sequential(*[self.encoder2_l1, self.activation_25, self.encoder2_l2, self.activation_25]).cuda()
        return [self.encoder2_l1, self.encoder2_l2]

    def encoder_forward(self, encoder_type, x):

        if encoder_type == 1:
            x = self.sigmoid(x)
            # x = self.activation(self.encoder_l1(x))
            # return self.activation(self.encoder_l2(x))
            return self.encoder1(x)

        else:
            # x = self.activation(self.encoder2_l1(x))
            # return self.activation(self.encoder2_l2(x))
            return self.encoder2(x)

    def create_classifier1(self):

        self.classifier1_l1 = nn.Linear(self.space_dim_1, self.num_classes)
        self.classifier1 = nn.Sequential(*[self.classifier1_l1]).cuda()
        return [self.classifier1_l1]

    def classifier1_forward(self, x):

        Classifier_logit = self.classifier1(x)  # self.classifier1_l1(x)
        Classifier_prob = self.sigmoid(Classifier_logit)
        return Classifier_logit, Classifier_prob

    def create_classifier2(self):

        self.classifier2_l1 = nn.Linear(self.space_dim_2, self.num_classes)
        self.classifier2 = nn.Sequential(*[self.classifier2_l1]).cuda()
        return [self.classifier2_l1]

    def classifier2_forward(self, x):

        Classifier_logit = self.classifier2(x)  # self.classifier2_l1(x)
        Classifier_prob = self.sigmoid(Classifier_logit)
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
        self.gen1_l1 = nn.Linear(self.x_dim + self.noise_dim,self.h_g_dim, bias=False)
        self.gen1_l2 = nn.Linear(self.h_g_dim,self.h_g_dim, bias=False)
        self.gen1_l3 = nn.Linear(self.h_g_dim, self.h_g_dim)
        self.gen1_l4 = nn.Linear(self.h_g_dim, self.y_dim)
        self.batch_norm1_g1 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        self.batch_norm2_g1 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        self.gen_ones = torch.ones(self.h_g_dim)
        self.gen_zeros = torch.zeros(self.h_g_dim)
        self.gen1 = nn.Sequential(*[self.gen1_l1, self.batch_norm1_g1, self.activation_2, self.gen1_l2, self.batch_norm2_g1, self.gen1_l3, self.activation_2, self.gen1_l4]).cuda()

        return [self.gen1_l1, self.gen1_l2, self.gen1_l3, self.gen1_l4]


    def gen1_forward(self, x, noise):
        noise = noise * 60
        inputs = torch.cat([x, noise], dim=1)
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
        return self.gen1(inputs)

    def create_gen2(self):
        self.gen2_l1 = nn.Linear(self.y_dim + self.noise_dim,self.h_g_dim, bias=False)
        self.gen2_l2 = nn.Linear(self.h_g_dim,self.h_g_dim, bias=False)
        self.gen2_l3 = nn.Linear(self.h_g_dim, self.h_g_dim)
        self.gen2_l4 = nn.Linear(self.h_g_dim, self.x_dim)
        self.batch_norm1_g2 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        self.batch_norm2_g2 = nn.BatchNorm1d(self.h_g_dim, eps=1e-6)
        self.gen_ones = torch.ones(self.h_g_dim)
        self.gen_zeros = torch.zeros(self.h_g_dim)
        self.gen2 = nn.Sequential(*[self.gen2_l1, self.batch_norm1_g2, self.activation_2, self.gen2_l2, self.batch_norm2_g2, self.gen2_l3, self.activation_2, self.gen2_l4]).cuda()

        return [self.gen2_l1, self.gen2_l2, self.gen2_l3, self.gen2_l4]

    def gen2_forward(self, x, noise):
        noise = noise * 60
        inputs = torch.cat([x, noise], dim=1)
        # x = self.gen2_l1(inputs)
        # x = self.batch_norm1(x)
        # x = self.gen_ones * x + self.gen_zeros
        # x = self.activation_2(x,0.2)
        # x = self.gen2_l2(x)
        # x = self.batch_norm2(x)
        # x = self.activation_2(self.gen2_l3(x), 0.2)
        # x = self.gen2_l4(x)
        # return x
        return self.gen2(inputs)

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


    def forward(self, subject_1, subject_2, labels, noise, eval_mode=False):

        generate_fake_z2 = self.gen1_forward(subject_1, noise)  # generate S1 -> S2
        generate_fake_z1 = self.gen2_forward(subject_2, noise) # generate S2 -> S1

        C_1_real_logit, C_1_real_prob = self.classifier1_forward(self.encoder_forward(1, subject_1))
        C_1_fake_logit, C_1_fake_prob = self.classifier1_forward(self.encoder_forward(1, generate_fake_z1))

        C_2_real_logit, C_2_real_prob = self.classifier2_forward(self.encoder_forward(2, subject_2))
        C_2_fake_logit, C_2_fake_prob = self.classifier2_forward(self.encoder_forward(2, generate_fake_z2))

        C_g_logit_real, C_g_prob_real = self.vrdn_forward(C_1_real_prob, C_2_real_prob)
        C_g_logit_rf, C_g_prob_rf = self.vrdn_forward(C_1_real_prob, C_2_fake_prob)
        C_g_logit_fr, C_g_prob_fr = self.vrdn_forward(C_1_fake_prob, C_2_real_prob)

        # calculate classiciation loss for separate Classififers
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
        G1_loss_dis = self.binary_criterion(D1_logit_fake, torch.ones_like(D1_logit_fake))
        G2_loss_dis = self.binary_criterion(D2_logit_fake, torch.ones_like(D2_logit_fake))
        G1_loss_fea = self.compute_generator_loss(generate_fake_z2, subject_2) #self.reconstruction_criterion(generate_fake_z2, subject_2) # generated feature similiarity
        G2_loss_fea = self.compute_generator_loss(generate_fake_z1, subject_1) #self.reconstruction_criterion(generate_fake_z1, subject_1) # generated feature similiarity
        G1_loss = G1_loss_dis + self.lamda_g_smi * G1_loss_fea
        G2_loss = G2_loss_dis + self.lamda_g_smi * G2_loss_fea

        if eval_mode:
            # compute accuracies for classifiers
            acc_c1_real = torch.mean(torch.eq(torch.argmax(C_1_real_logit, 1), torch.argmax(labels, 1)).float())
            acc_c1_fake = torch.mean(torch.eq(torch.argmax(C_1_fake_logit, 1), torch.argmax(labels, 1)).float())
            acc_c2_real = torch.mean(torch.eq(torch.argmax(C_2_real_logit, 1), torch.argmax(labels, 1)).float())
            acc_c2_fake = torch.mean(torch.eq(torch.argmax(C_2_fake_logit, 1), torch.argmax(labels, 1)).float())

            # Evaluate on testing stage in 3 settings: (1) real Z1 real Z2 (2) real Z1 fake Z2 (3) fake Z1 real Z2
            C_r1r2_logit, C_r1r2_prob = self.vrdn_forward(C_1_real_prob, C_2_real_prob) # real z1 + real z2
            acc_te_r1r2 = torch.mean(torch.eq(torch.argmax(C_r1r2_prob, 1), torch.argmax(labels, 1)).float())

            C_r1f2_logit, C_r1f2_prob = self.vrdn_forward(C_1_real_prob, C_2_fake_prob) # real z1 + fake z2
            acc_te_r1f2 = torch.mean(torch.eq(torch.argmax(C_r1f2_prob, 1), torch.argmax(labels, 1)).float())

            C_f1r2_logit, C_f1r2_prob = self.vrdn_forward(C_1_fake_prob, C_2_real_prob) # fake z1 + real z2
            acc_te_f1r2 = torch.mean(torch.eq(torch.argmax(C_f1r2_prob, 1), torch.argmax(labels, 1)).float())

            # first is rgb-d, rgb, depth => s1-s2, s1, s2
            return acc_te_r1r2, acc_te_r1f2, acc_te_f1r2, acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake

        return G1_loss, G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss

    def compute_generator_loss(self, fake, real_subj):

        return torch.mean(torch.norm(fake-real_subj, p=None))


    def compute_classifier_loss(self, real_logits, fake_logits, labels):

        return self.c_rf * torch.mean(torch.square(real_logits - labels)) + (1.0 - self.c_rf) * torch.mean(torch.square(fake_logits - labels))
        #return self.c_rf * self.gans_criterion(real_logits, labels) + (1.0 - self.c_rf) * self.gans_criterion(fake_logits, labels)


    def train_(self, epochs, train_loader, test_loader=None, out_dir=''):

        best_val_acc = 0.0
        module_names = [self.encoder1, self.encoder2, self.classifier1, self.classifier2, self.desc1_1, self.desc1_2,
                        self.desc2_1, self.desc2_2, self.vrdn, self.gen1, self.gen2]
        module_str = ['encoder_1', 'encoder_2', 'classifier_1', 'classifier_2', 'desc1_1',
                      'desc1_2', 'desc2_1', 'desc2_2', 'vrdn', 'gen1', 'gen2']
        optimizer_list = []

        # # define optimizers
        for module, m_str in zip(module_names, module_str):
            if m_str == 'gen1':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=3e-5))
            elif m_str == 'gen2':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=4e-5))
            else:
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=self.lr))

        for epoch in range(epochs):

            # G1_loss, G2_loss, D1_loss, D2_loss, E_1_loss, E_2_loss, C_g_loss_sum = [], [], [], [], [], [], []

            train_x, train_y, labels = train_loader.train_next_batch(self.batch_size)
            # for i, (train_x, train_y, labels) in tqdm(enumerate(train_loader)):

            for op in optimizer_list:
                op.zero_grad()

            noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim), dtype=torch.float)
            g1_l, g2_l, d1_l, d2_l, e1_l, e2_l, cg_l, c_1_l, c_2_l = self.forward(train_x.to(self.device), train_y.to(self.device),
                                                                    labels.to(self.device), noise_sample.to(device))

            print('Iteration = ', epoch, '  Loss_G_1 = %.4f' % g1_l.item(),' Loss_G_2 = %.4f' % g2_l.item(),
              '  Loss_D_1 = %.4f' % d1_l.item(),'  Loss_D_2 = %.4f' % d2_l.item(),' Loss_E_1 = %.4f' % e1_l.item(),
              '  Loss_E_2 = %.4f' % e2_l.item(),'  Loss_VRDCN = %.4f' % cg_l.item(), '  Loss_C_1 = %.4f' % c_1_l.item(),'  Loss_C_2 = %.4f' % c_2_l.item())

            # for idx in range(5):
            #     optimizer_list[9].zero_grad()
            #     optimizer_list[10].zero_grad()
            #     train_x, train_y, _ = train_loader.train_next_batch(self.batch_size)
            #     noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim), dtype=torch.float)
            #     generate_fake_z2 = self.gen1_forward(train_x.to(device), noise_sample.to(device))
            #     generate_fake_z1 = self.gen2_forward(train_y.to(device), noise_sample.to(device))
            #     g1_l = self.reconstruction_criterion(generate_fake_z2, train_y.to(device))  # generated feature similiarity
            #     g2_l = self.reconstruction_criterion(generate_fake_z1, train_x.to(device))  # generated feature similiarity

                # g1_l.backward(retain_graph=True)
                # g2_l.backward(retain_graph=True)
                # optimizer_list[9].step()
                # optimizer_list[10].step()
            g1_l.backward(retain_graph=True)
            g2_l.backward(retain_graph=True)
            d1_l.backward(retain_graph=True)
            d2_l.backward(retain_graph=True)
            e1_l.backward(retain_graph=True)
            e2_l.backward(retain_graph=True)
            cg_l.backward(retain_graph=True)
            c_1_l.backward(retain_graph=True)
            c_2_l.backward(retain_graph=True)

            for op in optimizer_list:
                op.step()

            if epoch % 50 == 0 and epoch != 0:

                with torch.no_grad():

                    test_x, test_y, test_labels = train_data.test_next_batch(train_data.sample_test_num)  # load all test samples


                    noise_sample = torch.tensor(self.sample_Noise(train_data.sample_test_num, self.noise_dim), dtype=torch.float)

                    sub_1_acc, sub_2_acc, sub_1_2_acc, acc_c1_r, acc_c1_f, acc_c2_r, acc_c2_f = self.forward(test_x.to(self.device), test_y.to(self.device),
                                                                     test_labels.to(self.device), noise_sample.to(device), eval_mode=True)
                    # first is rgb-d, rgb, depth => s1-s2, s1, s2
                    print('Iteration = ', epoch, '  Accuracy:  Subject_1_2 = %.4f' % sub_1_acc, ' Subject_1 = %.4f' % sub_2_acc,
                          '  Subject_2 = %.4f' % sub_1_2_acc, ' C_1_R = %.4f' % acc_c1_r, ' C_1_F = %.4f' % acc_c1_f, ' C_2_R = %.4f' % acc_c2_r, ' C_2_F = %.4f' % acc_c2_f)

                    # if sub_1_2_acc > best_val_acc:
                    #     best_val_acc = np.mean(sub_1_2_acc)
                    #     best_epoch = epoch
                    #     for op, module, name in zip(optimizer_list, module_names, module_str):
                    #         torch.save({
                    #             'epoch': best_epoch,
                    #             'model_state_dict': module.state_dict(),
                    #             'optimizer_state_dict': op.state_dict(),
                    #         }, os.path.join(out_dir, name))



seg=50
max_epochs=300001
batch_size = 64
learning_rate = 1e-5
eval_batch_size = 64

database = 'DHA' # assign evaluation dataset 'UWA30' and 'DHA'

# assign action number for different datasets
if database == 'UWA30':
    class_num = 30
elif database == 'UCB':
    class_num = 11
elif database == 'DHA':
    class_num = 23
else:
    class_num = 60

num_classes = class_num
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
out_path = '/home/ahmed/Desktop/model_experiments/pytorch_cyclegan_ours'

train_data = dataset_loader.data_loader(database)
train_data.read_train()

model = cyclegan(num_classes, batch_size, learning_rate, device, seg=seg).to(device)
# if not os.path.exists(out_path):
#     os.mkdir(out_path)

model.train_(max_epochs, train_data, out_dir=out_path)

