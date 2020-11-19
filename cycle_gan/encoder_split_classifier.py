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

class LSTM_Encoders(nn.Module):

    def __init__(self, input_size,hidden_size, num_layers=2):
        super(LSTM_Encoders, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
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

        self.epsilon = 1e-6
        self.c_rf = 0.99  # weights between real and fake
        self.cg_rf = 0.9  # the weights of the Cg() model (looks like =1 is more reasonable)
        self.hcg_dim = 100  # graph classifier Cg() hidden layer dimension for VRDCN
        self.lamda_g_smi = 0.1  # similarity weights
        self.lbd_t = 0.001  # triplet loss weight

        self.leaky_relu = nn.LeakyReLU(0.25)

        self.activation_25 = activation_2(0.25)
        self.activation_2 = activation_2()

        self.lstm_encoder = LSTM_Encoders(75, self.space_dim_1, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.softmax_criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.L1Loss(reduction='sum')
        self.triplet_criterion = HardTripletLoss()

        self.create_network()
        self.device = device

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def sample_Noise(self, m, n):
        return np.random.uniform(0., 1., size=[m, n])

    def create_encoder_1(self):

        self.encoder1 = nn.Sequential(*[self.lstm_encoder]).cuda()
        return None

    def create_encoder_2(self):
        self.encoder2 = nn.Sequential(*[self.lstm_encoder]).cuda()
        return None

    def encoder_forward(self, encoder_type, x):

        x = x.view(self.batch_size, self.seg, 75)

        if encoder_type == 1:
            x = self.encoder1(x)
        else:
            x = self.encoder2(x)
        x = x[:, -1, :]
        return x

    def create_classifier1(self):

        self.classifier1_l1 = nn.Linear(self.space_dim_1*2, self.num_classes)
        self.classifier1 = nn.Sequential(*[self.classifier1_l1]).cuda()
        return [self.classifier1_l1]

    def classifier1_forward(self, x):

        Classifier_logit = self.classifier1(x)
        Classifier_prob = self.softmax(Classifier_logit)
        return Classifier_logit, Classifier_prob

    def create_classifier2(self):

        self.classifier2_l1 = nn.Linear(self.space_dim_2*2, self.num_classes)
        self.classifier2 = nn.Sequential(*[self.classifier2_l1]).cuda()
        return [self.classifier2_l1]

    def classifier2_forward(self, x):

        Classifier_logit = self.classifier2(x)
        Classifier_prob = self.softmax(Classifier_logit)
        return Classifier_logit, Classifier_prob

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
        Classifier_g_prob = self.softmax(Classifier_g_logit)

        return Classifier_g_logit, Classifier_g_prob


    def create_network(self):

        enc1_vars = self.create_encoder_1()
        enc2_vars = self.create_encoder_2()
        vrdn_vars = self.create_vrdn()
        classifier1_vars = self.create_classifier1()
        classifier2_vars = self.create_classifier2()

        self.encoder1.apply(self.xavier_init)
        self.encoder2.apply(self.xavier_init)
        self.classifier1.apply(self.xavier_init)
        self.classifier2.apply(self.xavier_init)
        self.vrdn.apply(self.xavier_init)

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

    def forward(self, subject_1, labels, class_total, class_correct, is_eval=False):

        C_1_real_logit, C_1_real_prob = self.classifier1_forward(self.encoder_forward(1, subject_1))
        C_2_real_logit, C_2_real_prob = self.classifier2_forward(self.encoder_forward(2, subject_1))
        C_g_logit_real, C_g_prob_real = self.vrdn_forward(C_1_real_prob, C_2_real_prob)

        # calculate classiciation loss for separate Classififers
        if is_eval:
            C_1_loss = self.softmax_criterion(C_1_real_logit, torch.argmax(labels, 1))
            C_2_loss = self.softmax_criterion(C_2_real_logit, torch.argmax(labels, 1))
        else:

            temp_labels = torch.argmax(labels, 1)
            C_1_loss = self.softmax_criterion(C_1_real_logit[temp_labels<=19], temp_labels[temp_labels<=19])
            C_2_loss = self.softmax_criterion(C_2_real_logit[temp_labels>19], temp_labels[temp_labels>19])

        # calculate classiciation loss for Correlation Classififer
        C_g_loss_real = self.softmax_criterion(C_g_logit_real, torch.argmax(labels, 1))
        C_g_loss_sum = C_g_loss_real

        triplet_loss_1 = self.triplet_criterion.forward(self.encoder_forward(1, subject_1), torch.argmax(labels, 1))
        triplet_loss_2 = self.triplet_criterion.forward(self.encoder_forward(2, subject_1), torch.argmax(labels, 1))

        E_1_loss = self.lbd_t * triplet_loss_1 + C_1_loss
        E_2_loss = self.lbd_t * triplet_loss_2 + C_2_loss

        # compute accuracies for classifiers
        acc_c1_real = torch.mean(torch.eq(torch.argmax(C_1_real_logit, 1), torch.argmax(labels, 1)).float())
        acc_c2_real = torch.mean(torch.eq(torch.argmax(C_2_real_logit, 1), torch.argmax(labels, 1)).float())

        # Evaluate on testing stage in 3 settings: (1) real Z1 real Z2
        C_r1r2_logit, C_r1r2_prob = self.vrdn_forward(C_1_real_prob, C_2_real_prob)  # real z1 + real z2
        acc_te_r1r2 = torch.mean(torch.eq(torch.argmax(C_r1r2_prob, 1), torch.argmax(labels, 1)).float())

        # calculate per class accuracy
        y_true = torch.argmax(labels, 1).detach().cpu().numpy()
        y_pred = torch.argmax(C_r1r2_prob, 1).detach().cpu().numpy()
        class_total, class_correct = self.class_accuracy(y_pred, y_true, class_total, class_correct)

        return acc_te_r1r2, acc_c1_real, acc_c2_real, E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss, C_2_loss, class_total,\
               class_correct



    def train_(self, epochs, train_loader=None, test_loader=None, out_dir='', is_eval=False):

        wandb.init(project="Cycle-Gan", reinit=True)

        best_val_acc = 0.0
        module_names = [self.encoder1, self.encoder2, self.vrdn, self.classifier1, self.classifier2]
        module_str = ['encoder_1', 'encoder_2', 'vrdn', 'classifier_1', 'classifier_2']

        # # define optimizers
        optimizer_list = []
        for module, m_str in zip(module_names, module_str):
            if m_str == 'vrdn':
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=1e-3))
            else:
                optimizer_list.append(torch.optim.Adam(module.parameters(), lr=self.lr))

        if not is_eval:

            for epoch in range(epochs):

                acc_te_r1r2_num, acc_c1_r_sum, acc_c2_r_sum = [], [], []
                E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss_sum, C_2_loss_sum = [], [], [], [], []

                class_correct = list(0. for i in range(60))
                class_total = list(0. for i in range(60))

                for i, (train_x, train_y, labels) in tqdm(enumerate(train_loader)):

                    for op in optimizer_list[:-2]:
                        op.zero_grad()

                    acc_te_r1r2, acc_c1_real, acc_c2_real, e1_l, e2_l, cg_l, c_1_l, \
                    c_2_l, class_total, class_correct = self.forward(train_x.to(self.device), labels.to(self.device),
                                                                     class_total, class_correct)

                    e1_l.backward(retain_graph=True)
                    e2_l.backward(retain_graph=True)
                    cg_l.backward(retain_graph=True)

                    acc_te_r1r2_num.append(acc_te_r1r2.item())
                    acc_c1_r_sum.append(acc_c1_real.item())
                    acc_c2_r_sum.append(acc_c2_real.item())
                    E_1_loss.append(e1_l.item())
                    E_2_loss.append(e2_l.item())
                    C_g_loss_sum.append(cg_l.item())
                    C_1_loss_sum.append(c_1_l.item())
                    C_2_loss_sum.append(c_2_l.item())

                    for op in optimizer_list[:-2]:
                        op.step()


                print('\n\n[TRAIN] Epoch = ', epoch, ' Loss_E_1 = %.4f' % np.mean(E_1_loss),
                      '  Loss_E_2 = %.4f' % np.mean(E_2_loss), '  Loss_VRDCN = %.4f' % np.mean(C_g_loss_sum),
                      '  Loss_C_1 = %.4f' % np.mean(C_1_loss_sum), '  Loss_C_2 = %.4f' % np.mean(C_2_loss_sum))

                print('\n[TRAIN] Epoch = ', epoch, '  Accuracy:  Subject_1_2 = %.4f' % np.mean(acc_te_r1r2_num),
                          ' C_1_R = %.4f' % np.mean(acc_c1_r_sum), ' C_2_R = %.4f' % np.mean(acc_c2_r_sum))

                log_dict = {'Train_E1': np.mean(E_1_loss), "Train_E2": np.mean(E_2_loss), 'Train_VRDN': np.mean(C_g_loss_sum),
                            'Train_C1': np.mean(C_1_loss_sum), "Train_C2": np.mean(C_2_loss_sum),
                            'Train_Subject_1_2': np.mean(acc_te_r1r2_num), 'Train_C_1_R': np.mean(acc_c1_r_sum),
                            'Train_C_2_R':np.mean(acc_c2_r_sum)}

                wandb.log(log_dict)
                _ = self.calculate_accuracy(class_total, class_correct, '[TRAIN]')

                if epoch % 5 == 0: #and epoch != 0:

                    with torch.no_grad():
                        class_correct_eval = list(0. for i in range(60))
                        class_total_eval = list(0. for i in range(60))

                        acc_te_r1r2_num, acc_c1_r_sum, acc_c2_r_sum = [], [], []
                        E_1_loss, E_2_loss, C_g_loss_sum, C_1_loss_sum, C_2_loss_sum = [], [], [], [], []

                        for test_x, test_y, test_labels in test_loader:

                            acc_te_r1r2, acc_c1_real, acc_c2_real, e1_l, e2_l, cg_l, c_1_l, c_2_l, class_total_eval, \
                            class_correct_eval = self.forward(test_x.to(self.device), test_labels.to(self.device),
                                                              class_total_eval, class_correct_eval, is_eval=True)

                            acc_te_r1r2_num.append(acc_te_r1r2.item())
                            acc_c1_r_sum.append(acc_c1_real.item())
                            acc_c2_r_sum.append(acc_c2_real.item())
                            E_1_loss.append(e1_l.item())
                            E_2_loss.append(e2_l.item())
                            C_g_loss_sum.append(cg_l.item())
                            C_1_loss_sum.append(c_1_l.item())
                            C_2_loss_sum.append(c_2_l.item())

                        print('\n [TEST] Epoch = ', epoch, '  Accuracy:  Subject_1_2 = %.4f' % np.mean(acc_te_r1r2_num),
                              ' C_1_R = %.4f' % np.mean(acc_c1_r_sum), ' C_2_R = %.4f' % np.mean(acc_c2_r_sum))

                        print('\n\n[TEST] Epoch = ', epoch, ' Loss_E_1 = %.4f' % np.mean(E_1_loss),
                              '  Loss_E_2 = %.4f' % np.mean(E_2_loss), '  Loss_VRDCN = %.4f' % np.mean(C_g_loss_sum),
                              '  Loss_C_1 = %.4f' % np.mean(C_1_loss_sum), '  Loss_C_2 = %.4f' % np.mean(C_2_loss_sum))

                        acc_dict = {'Test_E1': np.mean(E_1_loss), "Test_E2": np.mean(E_2_loss), 'Test_VRDN': np.mean(C_g_loss_sum),
                            'Test_C1': np.mean(C_1_loss_sum), "Test_C2": np.mean(C_2_loss_sum), 'Test_Subject_1_2': np.mean(acc_te_r1r2_num),
                                    'Test_C_1_R': np.mean(acc_c1_r_sum), 'Test_C_2_R':np.mean(acc_c2_r_sum)}

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
        else:
            print('Loading Model from checkpoint')
            for name, module in zip(module_str[:-2], module_names[:-2]):
                checkpoint = torch.load(os.path.join(out_dir, name))
                module.load_state_dict(checkpoint['model_state_dict'])
            print('Weights loaded from {}'.format(out_dir))

            visual = Draw3DSkeleton(save_path='./temp')
            counter = 0
            unique = []
            with torch.no_grad():
                for train_x, train_y, labels in test_loader:
                    cur_label = torch.argmax(labels)
                    if cur_label not in unique:
                        unique.append(cur_label)
                        train_x = train_x.view(1, self.seg, 75)
                        reconstructed_img = self.gen1(train_x.to(self.device))
                        print('Loss for class {} is {}'.format(cur_label, self.reconstruction_criterion(reconstructed_img,
                                                                                                        train_x.to(self.device))))
                    else:
                        if len(unique) == 60:
                            break
                        continue
                    # visual.visual_skeleton_batch(reconstructed_img.detach().cpu().numpy(),
                    #                              torch.argmax(labels, 1).numpy(), self.seg, '{}'.format(counter))


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

seg=25
max_epochs=300
batch_size = 1024
learning_rate = 1e-4
eval_batch_size = 1024
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
out_path = '/home/ahmed/Desktop/model_experiments/encoder_split_classifier'
if not os.path.exists(out_path):
    os.mkdir(out_path)

train_path ='/home/ahmed/Desktop/datasets/skeleton_dataset/cross_subject_data/trans_train_data.pkl'
test_path = '/home/ahmed/Desktop/datasets/skeleton_dataset/cross_subject_data/trans_test_data.pkl'
train_dataset = custom_dataloaders.pytorch_dataloader(batch_size, train_path=train_path, seg=seg)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
eval_dataset = custom_dataloaders.pytorch_dataloader(eval_batch_size, test_path=test_path, is_train=False, seg=seg)
eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=True, **kwargs)

model = cyclegan(num_classes, batch_size, learning_rate, device, seg=seg).to(device)
model.train_(max_epochs, train_loader, eval_loader, out_dir=out_path)
# model.train_(max_epochs, test_loader=eval_loader, out_dir=out_path, is_eval=True)