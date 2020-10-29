import time
import wandb
import logging
import datetime
import argparse
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_source_reader_video import SkeletonsDataset, SimpleDataset, FolderDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from embeddings.main_model import *
from ntu_preprocess import *
import cv2
import pickle
import sys
from embeddings.main_model import resnet18_train
from models.resnet_models import resnet18
from models.inceptionresnet import InceptionResnetV1
from center_loss import CenterLoss

# os.environ["WANDB_MODE"] = "dryrun"
os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"


def train():
    '''
        Function for training 1 epoch of network
        args:
            -draw_graph: argument for whether to draw graph displaying number of frames in each example or not of a single batch
        returns:
            -current loss value
            -training accuracy
    '''

    class_correct = list(0. for i in range(60))
    class_total = list(0. for i in range(60))

    model.train()  # Turn on the train mode
    total_loss = []
    start_time = time.time()
    sum_curl_loss = 0
    batch_count = 0
    log_interval = 500
    sum_skel_loss = []
    sum_video_loss = []

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device)
        targets = targets.to(device)
        targets = targets.view(-1)
        # skel_data = data
        # skel_data = data
        # skel_decoded, video_decoded = model(data)
        x_feat, skel_decoded = model(data)
        c_loss = criterion_cent(x_feat, targets)
        loss = skel_criterion(skel_decoded, targets)

        # Calculating accuracies
        _, predicted = torch.max(skel_decoded.view(-1, len(classes)), 1)
        c = (predicted == targets).squeeze()

        for i in range(targets.shape[0]):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] = class_total[label] + 1

        # video_loss = video_criterion(video_decoded, skel_data)
        # sum_skel_loss.append(skel_loss.item())
        # sum_video_loss.append(video_loss.item())
        # loss = skel_loss #+ video_loss
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()
        loss = loss + c_loss
        loss.backward()
        optimizer.step()
        optimizer_centloss.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        total_loss.append(loss.item() + c_loss.item())
        # sum_curl_loss += loss.item()

        # logging interval
        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:8d}/{:8d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | center_loss {:5.2f} '.format(
                epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, loss.item(), c_loss.item()))
            start_time = time.time()

        # wandb.sklearn.plot_confusion_matrix(targets.cpu().numpy().squeeze(), predicted.cpu().numpy(), labels=classes)

    # print(" Mean Skel LOSS : {} and STD DEV : {}".format(np.mean(sum_skel_loss), np.std(sum_skel_loss)))
    # print(" Mean Video LOSS : {} and STD DEV : {}".format(np.mean(sum_video_loss), np.std(sum_video_loss)))
    epoch_loss = np.sum(total_loss) / len(train_loader)

    print('TRAINING Accuracies now !')
    print('=' * 89)
    acc = calculate_accuracy(class_correct, class_total)
    print('=' * 89)
    return epoch_loss, acc
    # write_to_graph('train/loss', sum_curl_loss/len(train_loader), writer, step_count_tb)


def evaluate(eval_model, eval_loader):
    '''
        Function to evaluate model performance
        args:
            -eval_model: model to be used for evaluation purposes
            -eval_loader: data loader for data loading
        returns:
            Loss and accuracy for given dataset
    '''

    class_correct = list(0. for i in range(60))
    class_total = list(0. for i in range(60))

    eval_model.eval()  # Turn on the evaluation mode
    total_loss = []

    with torch.no_grad():
        for data, targets in eval_loader:

            data = data.to(device)  # 100, 75, 150
            targets = targets.to(device)
            targets = targets.view(-1)
            # skel_data = data#.view(75 * batch_size, 150) # 7500, 150
            # data = data.reshape((100,1,11250))
            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
            # skel_decoded, video_decoded = model(data)
            x_feat, skel_decoded = model(data)
            c_loss = criterion_cent(x_feat, targets)
            loss = skel_criterion(skel_decoded, targets)
            # calculate all the losses and perform backprop
            # loss = skel_loss #+ video_loss
            total_loss.append(loss.item() + c_loss.item())

            # Calculating accuracies code starts here
            _, predicted = torch.max(skel_decoded.view(-1, len(classes)), 1)
            c = (predicted == targets).squeeze()

            for i in range(targets.shape[0]):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1
            # Calculating accuracies code ends here

    print('TEST Accuracies now !')
    print('=' * 89)
    val_acc = calculate_accuracy(class_correct, class_total)

    return np.sum(total_loss) / len(eval_loader), val_acc


def calculate_accuracy(class_correct, class_total):
    acc_sum = 0
    for i in range(len(classes)):
        print(class_total[i])
        class_accuracy = 100 * class_correct[i] / class_total[i]
        # error_rate = 100 - class_accuracy
        acc_sum += class_accuracy
        print('%d Accuracy of %5s : %2d %% and total count : %2d ' % (i, classes[i], class_accuracy, class_total[i]))
    print('=' * 89)
    print('Mean Average Accuracy of Camera View : %2f %%' % (acc_sum / 60))
    print('=' * 89)

    return acc_sum / len(classes)  # , error_rate
    # write_to_graph(split+'/Accuracy', acc_sum/60, writer, epoch)


# training arguments
parser = argparse.ArgumentParser(description="Skeleton Classification Training Script")
parser.add_argument("-lr", "--learning_rate", default=0.1, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=100, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=100, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='/home/ahmed/Desktop/dataset_skeleton/trans_best_data/train',
                    help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='/home/ahmed/Desktop/dataset_skeleton/trans_best_data/test',
                    help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-hid_dim", "--nhid", type=int, default=8, help='Number of hidden dimenstions, default is 100')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--train", help="Put Model in training mode", default=True)
parser.add_argument("-f", "--frame_count", help="Frame count per video", default=60)
parser.add_argument("-c", "--checkpoint", help="path to store model weights",
                    default="/home/ahmed/Desktop/model_experiments/inception_resnet")
parser.add_argument("-bs", "--batch_shuffle", help="path to store model weights", default=True)
parser.add_argument("-rc", "--resume_checkpoint", help="path to store model weights",
                    default="./logs/output_2020-09-22 12:37:39.576905/epoch_2020-09-23 06:34:34.803320")
parser.add_argument("-r", "--resume_bool", default=False, help='Whether to resume training or start from scratch')
parser.add_argument("-ac", "--ae_checkpoint", help="path to load autoencoder from",
                    default="./autoencoder_weights/subject_video_ae_int_rgb.pth")

args = parser.parse_args()

classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up",
           "clapping", "reading", "writing"
    , "tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses",
           "take off glasses", "put on a hat/cap",
           "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping",
           "jump up", "phone call", "play with phone/tablet",
           "type on a keyboard", "point to something", "taking a selfie", "check time (from watch)", "rub two hands",
           "nod head/bow", "shake head", "wipe face",
           "salute", "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down",
           "headache", "chest pain", "back pain", "neck pain",
           "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing", "pat on back", "point finger", "hugging",
           "giving object", "touch pocket", "shaking hands",
           "walking towards", "walking apart"]

# initalize wandb
wandb.init(project="Cycle-Gan", reinit=True)
# Set up a specific logger with our desired output level

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

if args.train:

    # base_path = "/home/hashmi/Desktop/dataset/NTURGBD-60_120/ntu_60"  # "/home/neuralnet/NW_UCLA/" #
    # train_dataset = SkeletonsDataset(base_path, image_dataset=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.batch_shuffle, **kwargs)
    # eval_dataset = SkeletonsDataset(base_path, mode='eval', image_dataset=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=args.batch_shuffle, **kwargs)
    # base_path = "autoencoder_features/"  # "/home/neuralnet/NW_UCLA/" #
    train_dataset = FolderDataset(args.train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.batch_shuffle, **kwargs)
    eval_dataset = FolderDataset(args.eval_data)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=args.batch_shuffle, **kwargs)
    # # '''
    #     Loading all the models here
    #     Classification Network
    #     Defining the criterion for losses
    # '''
    # model = UnsuperVisedAE(batch_size=args.batch_size).to(device)
    # model = SkeletonAutoEnoder().to(device)
    # model = VideoAutoEnoder_sep(batch_size=args.batch_size).to(device)
    # model = skeleton_lstm(n_features=150).to(device)
    # model = classification_nn(num_feature=128, num_class=60).to(device)
    # model = resnet18().to(device)
    model = InceptionResnetV1(classify=True, num_classes=60, num_channels=3).to(device)
    # model = resnet18_train().to(device)
    print(model)
    skel_criterion = nn.CrossEntropyLoss()  # nn.MSELoss(reduction='sum')
    criterion_cent = CenterLoss(num_classes=60, feat_dim=128)

    # video_criterion = nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)  #
    optimizer_centloss = torch.optim.RMSprop(criterion_cent.parameters(), lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, min_lr=0.000001)

    best_val_loss = float("inf")
    max_epochs = args.epochs  # number of epochs
    step_count_tb = 1  # xaxis for calculating loss value for tensorboard

    # Saving and writing model as a state_dict
    # Training procedure starts
    current_date_time = str(datetime.datetime.now()).split(",")[0]
    create_dir(args.checkpoint)
    output_path = os.path.join(args.checkpoint, "output_" + str(current_date_time))
    create_dir(output_path)  # creating the directory where epochs will be saved
    # skeleton_output = os.path.join(output_path, 'skeleton_diagrams')

    '''
    Resuming from the specific check point
    '''
    if args.resume_bool:
        # Resuming the model from the specific checkpoint
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    lr_change_count = 0
    epoch_output_path = output_path + "/epoch_" + str(current_date_time)
    for epoch in range(0, max_epochs):

        epoch_start_time = time.time()
        current_date_time = str(datetime.datetime.now()).split(",")[0]

        tr_loss, tr_acc = train()
        print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.4f} | '.format(epoch,
                                                                                     (time.time() - epoch_start_time),
                                                                                     tr_loss))
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.4f} | '.format(epoch, (
                time.time() - epoch_start_time), tr_loss))

        val_loss, val_acc = evaluate(model, eval_loader)
        scheduler.step(val_loss)

        wandb.log({"train_loss": tr_loss, "train_accuracy": tr_acc, "test_loss": val_loss,
                   "learing_rate": optimizer.param_groups[0]['lr'], "test_accuracy": val_acc})
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | valid acc {:5.4f} | '.format(epoch,
                                                                                                         (
                                                                                                                     time.time() - epoch_start_time),
                                                                                                         val_loss,
                                                                                                         val_acc))
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (
                time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # write_to_graph('Val/loss', val_loss, writer, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, epoch_output_path)

# else:
#
#     test_dataset = SkeletonsDataset(args.test_data, args.batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
#     # loading the model again to evaluate for the test set.
#     checkpoint = torch.load(args.checkpoint)
#     model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     test_loss, test_acc = evaluate(model, test_loader, draw_img=True, visualize=False, out_path='./visual')
#     print('=' * 89)
#     print('| End of testing | test loss {:5.2f} | test ppl {:8.2f} | test acc {:8.2f}'.format(
#         test_loss, math.exp(test_loss), test_acc))
#     print('=' * 89)
