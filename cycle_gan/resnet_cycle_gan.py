import os
import wandb
import torch
import functools
import numpy as np
from torch import nn
from tqdm import tqdm
import custom_dataloaders
import torch.nn.functional as F
from torch.utils.data import DataLoader
from online_triplet_loss.losses import *

# os.environ["WANDB_MODE"] = "dryrun"
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


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 1
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Identity(nn.Module):
    def forward(self, x):
        return x


class cyclegan(nn.Module):

    def __init__(self, num_classes, batch_size, learning_rate, device, seg=30):

        super(cyclegan, self).__init__()

        self.batch_size = batch_size
        self.lr = learning_rate
        self.x_dim = seg*75  # seg * 75
        self.y_dim = seg*75  # seg * 75
        self.num_classes = num_classes
        self.hx_dim = 800  #  Encoder 1 hidden dimension
        self.hy_dim = 800  #  Encoder 2 hidden dimension
        self.space_dim_1 = 300  # 256 # First Subspace dimension
        self.space_dim_2 = 300  # 256 # Second Subspace dimension

        self.noise_dim = 300  # 256 # Noise dimension same as the sub space dimensions
        self.epsilon = 1e-6

        self.hd1_dim = 400  # 256 # First Discriminator hidden dimension
        self.hd2_dim = 400  # 256 # Second Discriminator hidden dimension

        # self.hg1_dim = 512
        # self.hg2_dim = 512

        self.h_g_dim = 500  # 512 # Only one Dim for both of the Generators

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

        self.triplet_criterion = HardTripletLoss(margin=1.0).cuda()

        # self.sigmoid_criterion = F.binary_cross_entropy_with_logits()
        # self.triplet_criterion = batch_hard_triplet_loss()

        ''' parameters for resnet generator and desc '''
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_idt = 0.5
        self.input_nc = 3
        self.ndf = 64



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

        self.encoder_l1 = nn.Linear(self.x_dim, self.hx_dim)
        self.encoder_l2 = nn.Linear(self.hx_dim, self.space_dim_1)
        self.encoder1 = nn.Sequential(
            *[self.encoder_l1, self.activation_25, self.encoder_l2, self.activation_25]).cuda()
        return [self.encoder_l1, self.encoder_l2]

    def create_encoder_2(self):
        self.encoder2_l1 = nn.Linear(self.y_dim, self.hy_dim)
        self.encoder2_l2 = nn.Linear(self.hy_dim, self.space_dim_2)
        self.encoder2 = nn.Sequential(
            *[self.encoder2_l1, self.activation_25, self.encoder2_l2, self.activation_25]).cuda()
        return [self.encoder2_l1, self.encoder2_l2]

    def encoder_forward(self, encoder_type, x):

        if encoder_type == 1:
            x = self.sigmoid(x)
            return self.encoder1(x)

        else:
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

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        desc1 = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.desc1 = nn.Sequential(*desc1)

        return ['shit']


    def minibatch1(self, previous_input, input, num_kernels=5, kernel_dim=3):

        activation = torch.reshape(input, (-1, num_kernels, kernel_dim))
        a = torch.unsqueeze(activation, 3)
        b = torch.unsqueeze(torch.transpose(activation, dim0=1, dim1=2), 0)
        b = b.permute(0, 3, 2, 1)
        diffs = a - b  # [1, 2, 0]), 0)
        abs_diffs = torch.sum(torch.abs(diffs), dim=2)
        minibatch_features = torch.sum(torch.exp(-abs_diffs), dim=2)
        return torch.cat([previous_input, minibatch_features], dim=1)

    def des1_forward(self, x):
        return self.desc1(x)

    def create_descr2(self):

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        desc2 = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.desc2 = nn.Sequential(*desc2)

        return ['shit']

    def des2_forward(self, x):
        return self.desc2(x)

    def create_gen1(self):

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.gen1 = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

        return ['shits']

    def create_gen2(self):

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.gen2 = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
        return ['shits']

    def gen_forward(self, real_A, real_B):

        fake_B = self.gen1(real_A)  # G_A(A)
        rec_A = self.gen2(fake_B)  # G_B(G_A(A))
        fake_A = self.gen2(real_B)  # G_B(B)
        rec_B = self.gen1(fake_A)  # G_A(G_B(B))

        return fake_B, rec_A, fake_A, rec_B

    def create_vrdn(self):

        self.vrdn_l1 = nn.Linear(self.num_classes * self.num_classes, self.hcg_dim)
        self.vrdn_l2 = nn.Linear(self.hcg_dim, self.num_classes)
        self.vrdn = nn.Sequential(*[self.vrdn_l1, self.activation_25, self.vrdn_l2]).cuda()

        return [self.vrdn_l1, self.vrdn_l2]

    def vrdn_forward(self, x, y):

        # make correlation matrix
        C_prob_1 = torch.unsqueeze(x, -1)
        C_prob_2 = torch.unsqueeze(y, 1)
        W_feature = torch.matmul(C_prob_1, C_prob_2)
        C_hw = torch.reshape(W_feature, [-1, self.num_classes * self.num_classes])
        # fordward pass
        Classifier_g_logit = self.vrdn(C_hw)
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
        self.desc1.apply(self.xavier_init)
        self.desc2.apply(self.xavier_init)


    def forward(self, subject_1, subject_2, labels, noise, eval_mode=False):

        # fake_B, rec_A, fake_A, rec_B
        #print(subject_1.shape, subject_2.shape, labels.shape)
        generate_fake_z2, rec_A, generate_fake_z1, rec_B = self.gen_forward(subject_1, subject_2)
        #print('Gen out shapes: ', generate_fake_z1.shape, rec_A.shape, generate_fake_z1.shape, rec_B.shape)

        generate_fake_z2 = generate_fake_z2[:, :, :, :25]
        generate_fake_z1 = generate_fake_z1[:, :, :, :25]
        rec_A = rec_A[:, :, :, :25]
        rec_B = rec_B[:, :, :, :25]
        #print(rec_A.shape)

        subject_1_flatten = subject_1.view(self.batch_size, -1)
        subject_2_flatten = subject_2.view(self.batch_size, -1)
        generate_fake_z1_flatten = generate_fake_z1.reshape(self.batch_size, 2250)
        generate_fake_z2_flatten = generate_fake_z2.reshape(self.batch_size, 2250)
        rec_A_flatten = rec_A.reshape(self.batch_size, 2250)
        rec_B_flatten = rec_B.reshape(self.batch_size, 2250)

        #print('After rehsape ',generate_fake_z1_flatten.shape, rec_A_flatten.shape, generate_fake_z2_flatten.shape,
       #       rec_B_flatten.shape)

        C_1_real_logit, C_1_real_prob = self.classifier1_forward(self.encoder_forward(1, subject_1_flatten))
        C_1_fake_logit, C_1_fake_prob = self.classifier1_forward(self.encoder_forward(1, generate_fake_z1_flatten))

        C_2_real_logit, C_2_real_prob = self.classifier2_forward(self.encoder_forward(2, subject_2_flatten))
        C_2_fake_logit, C_2_fake_prob = self.classifier2_forward(self.encoder_forward(2, generate_fake_z2_flatten))

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

        temp_1 = self.encoder_forward(1, subject_1_flatten)
        triplet_loss_1 = self.triplet_criterion(temp_1, torch.argmax(labels, 1))

        temp_2 = self.encoder_forward(2, subject_2_flatten)
        triplet_loss_2 = self.triplet_criterion.forward(temp_2, torch.argmax(labels, 1))

        # print('Triplet loss: ', triplet_loss_1.item(), triplet_loss_2.item())
        E_1_loss = self.lbd_t * triplet_loss_1 + C_1_loss
        E_2_loss = self.lbd_t * triplet_loss_2 + C_2_loss

        ''' Backward for Generator'''
        # Identity loss
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.gen1(subject_2)
            idt_A = idt_A[:, :, :, :25]
            loss_idt_A = self.criterionIdt(idt_A, subject_2) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.gen2(subject_1)
            idt_B = idt_B[:, :, :, :25]
            loss_idt_B = self.criterionIdt(idt_B, subject_1) * self.lambda_A * self.lambda_idt

        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        desc1_out = self.desc1(generate_fake_z2)
        loss_G_A = self.criterionGAN(desc1_out, torch.ones_like(desc1_out))
        # GAN loss D_B(G_B(B))
        desc2_out = self.desc2(generate_fake_z1)
        loss_G_B = self.criterionGAN(desc2_out, torch.ones_like(desc2_out))
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, subject_1) * self.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, subject_2) * self.lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        ''' Backward for Discriminator'''
        pred_real = self.des1_forward(subject_2)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        # Fake
        pred_fake = self.des1_forward(generate_fake_z2)
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        # Combined loss and calculate gradients
        loss_D1 = (loss_D_real + loss_D_fake) * 0.5

        pred_real = self.des2_forward(subject_1)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        # Fake
        pred_fake = self.des2_forward(generate_fake_z1)
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        # Combined loss and calculate gradients
        loss_D2 = (loss_D_real + loss_D_fake) * 0.5

        if eval_mode:
            # compute accuracies for classifiers
            acc_c1_real = torch.mean(torch.eq(torch.argmax(C_1_real_logit, 1), torch.argmax(labels, 1)).float())
            acc_c1_fake = torch.mean(torch.eq(torch.argmax(C_1_fake_logit, 1), torch.argmax(labels, 1)).float())
            acc_c2_real = torch.mean(torch.eq(torch.argmax(C_2_real_logit, 1), torch.argmax(labels, 1)).float())
            acc_c2_fake = torch.mean(torch.eq(torch.argmax(C_2_fake_logit, 1), torch.argmax(labels, 1)).float())

            # Evaluate on testing stage in 3 settings: (1) real Z1 real Z2 (2) real Z1 fake Z2 (3) fake Z1 real Z2
            C_r1r2_logit, C_r1r2_prob = self.vrdn_forward(C_1_real_prob, C_2_real_prob)  # real z1 + real z2
            acc_te_r1r2 = torch.mean(torch.eq(torch.argmax(C_r1r2_prob, 1), torch.argmax(labels, 1)).float())

            C_r1f2_logit, C_r1f2_prob = self.vrdn_forward(C_1_real_prob, C_2_fake_prob)  # real z1 + fake z2
            acc_te_r1f2 = torch.mean(torch.eq(torch.argmax(C_r1f2_prob, 1), torch.argmax(labels, 1)).float())

            C_f1r2_logit, C_f1r2_prob = self.vrdn_forward(C_1_fake_prob, C_2_real_prob)  # fake z1 + real z2
            acc_te_f1r2 = torch.mean(torch.eq(torch.argmax(C_f1r2_prob, 1), torch.argmax(labels, 1)).float())

            # first is rgb-d, rgb, depth => s1-s2, s1, s2
            return [loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, loss_D1, loss_D2,
                E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss, acc_te_r1r2, acc_te_r1f2, acc_te_f1r2,
                    acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake]

        loss_G.backward(retain_graph=True)
        loss_D1.backward(retain_graph=True)
        loss_D2.backward(retain_graph=True)
        E_1_loss.backward(retain_graph=True)
        E_2_loss.backward(retain_graph=True)
        C_g_loss_sum.backward(retain_graph=True)
        C_1_loss.backward(retain_graph=True)
        C_2_loss.backward(retain_graph=True)

        return [loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, loss_D1, loss_D2,
                E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss]

    def compute_generator_loss(self, fake, real_subj):

        return torch.mean(torch.norm(fake - real_subj, p=None))

    def compute_classifier_loss(self, real_logits, fake_logits, labels):

        return self.c_rf * torch.mean(torch.square(real_logits - labels)) + (1.0 - self.c_rf) * torch.mean(
            torch.square(fake_logits - labels))


    def print_log(self, epoch, total_batches, epoch_loss, eval_=False):

        log_dict = {}
        if eval_:

            str_names = ['loss_G', 'loss_G_A', 'loss_G_B', 'loss_cycle_A', 'loss_cycle_B', 'loss_idt_A', 'loss_idt_B', 'loss_D1', 'loss_D2',
             'E_1_loss', 'E_2_loss', 'C_g_loss_sum', 'C_1_loss', 'C_2_loss', 'sub_1_2', 'sub_1_acc', 'sub_2_acc',
             'acc_c1_real', 'acc_c1_fake', 'acc_c2_real', 'acc_c2_fake']

            print_str = '[TEST] Epoch = ' + str(epoch) + ' '
            for loss, name in zip(epoch_loss, str_names):
                print_str = print_str + name + ' = ' + str(loss / total_batches) + ' '
                log_dict['test ' + name] = loss

        else:
            str_names = ['G_L_C', 'loss_G_A', 'loss_G_B', 'loss_cycle_A', 'loss_cycle_B', 'loss_idt_A', 'loss_idt_B',
                        'loss_D1', 'loss_D2', 'E_1_loss', 'E_2_loss', 'C_g_loss_sum', 'C_1_loss', 'C_2_loss']

            print_str = '[TRAIN] Epoch = ' + str(epoch) + ' '
            for loss, name in zip(epoch_loss, str_names):
                print_str = print_str + name + ' = ' + str(loss/total_batches) + ' '
                log_dict['train '+name] = loss

        print(print_str)
        return log_dict

    def train_(self, epochs, train_loader, test_loader=None, out_dir=''):

        wandb.init(project="Cycle-Gan", reinit=True)

        best_val_acc = 0.0
        module_names = [self.encoder1, self.encoder2, self.classifier1, self.classifier2, self.desc1, self.desc2,
                        self.vrdn, self.gen1, self.gen2]
        module_str = ['encoder_1', 'encoder_2', 'classifier_1', 'classifier_2', 'desc1',
                      'desc2', 'vrdn', 'gen1', 'gen2']
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

            epoch_loss = np.zeros(14)
            for i, (train_x, train_y, labels) in tqdm(enumerate(train_loader)):

                for op in optimizer_list:
                    op.zero_grad()

                noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim), dtype=torch.float)
                loss = self.forward(train_x.to(self.device), train_y.to(self.device), labels.to(self.device),
                                                                                      noise_sample.to(device))
                ''' loss backward and optimizer step '''
                for i in range(len(loss)):
                    epoch_loss[i] += loss[i].item()

                for op in optimizer_list:
                    op.step()

            if epoch % 50 == 0 and epoch != 0:

                with torch.no_grad():
                    epoch_loss_v = np.zeros(21)
                    acc_te_r1r2_num, acc_te_r1f2_num, acc_te_f1r2_num, acc_c1_r_sum, acc_c1_f_sum, acc_c2_r_sum, acc_c2_f_sum = [], [], [], [], [], [], []
                    for test_x, test_y, test_labels in test_loader:
                        # test_labels = torch.argmax(test_labels, 1)
                        noise_sample = torch.tensor(self.sample_Noise(self.batch_size, self.noise_dim),
                                                    dtype=torch.float)

                        epoch_loss_val = self.forward(test_x.to(self.device), test_y.to(self.device),
                            test_labels.to(self.device), noise_sample.to(device), eval_mode=True)

                        for i in range(len(epoch_loss_val)):
                            epoch_loss_v[i] += epoch_loss_val[i].item()

                    wandb.log(self.print_log(epoch, len(test_loader), epoch_loss_v, eval_=True))

                    if np.mean(acc_te_r1r2_num) > best_val_acc:
                        best_val_acc = np.mean(acc_te_r1r2_num)
                        best_epoch = epoch
                        for op, module, name in zip(optimizer_list, module_names, module_str):
                            torch.save({
                                'epoch': best_epoch,
                                'model_state_dict': module.state_dict(),
                                'optimizer_state_dict': op.state_dict(),
                            }, os.path.join(out_dir, name))
            wandb.log(self.print_log(epoch, len(train_loader), epoch_loss))


classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up",
           "clapping", "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe",
           "take off a shoe", "put on glasses",
           "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
           "reach into pocket", "hopping",
           "jump up", "phone call", "play with phone/tablet", "type on a keyboard", "point to something",
           "taking a selfie", "check time (from watch)", "rub two hands",
           "nod head/bow", "shake head", "wipe face", "salute", "put palms together", "cross hands in front",
           "sneeze/cough", "staggering", "falling down",
           "headache", "chest pain", "back pain", "neck pain", "nausea/vomiting", "fan self", "punch/slap", "kicking",
           "pushing", "pat on back", "point finger", "hugging",
           "giving object", "touch pocket", "shaking hands", "walking towards", "walking apart"]

seg = 30
max_epochs = 1000
batch_size = 32
learning_rate = 1e-4
eval_batch_size = 32
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
out_path = '/home/ahmed/Desktop/model_experiments/pytorch_cyclegan_3_3'
if not os.path.exists(out_path):
    os.mkdir(out_path)

train_path = '/home/ahmed/Desktop/datasets/skeleton_dataset/cross_subject_data/trans_train_data.pkl'
test_path = '/home/ahmed/Desktop/datasets/skeleton_dataset/cross_subject_data/trans_test_data.pkl'

train_dataset = custom_dataloaders.pytorch_dataloader(batch_size, train_path=train_path, seg=seg, reshape=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
eval_dataset = custom_dataloaders.pytorch_dataloader(batch_size, test_path=test_path, is_train=False, seg=seg, reshape=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)


model = cyclegan(num_classes, batch_size, learning_rate, device, seg=seg).to(device)
model.train_(max_epochs, train_loader, eval_loader, out_dir=out_path)


