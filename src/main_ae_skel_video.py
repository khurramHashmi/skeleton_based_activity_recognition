import time
import datetime
import wandb
import argparse
from utils import *
from model_transformer import *
from data_source_reader_video import SkeletonsDataset
from torch.utils.data import DataLoader
from embeddings.main_model import classification_network_128
import torch.nn as nn

class_correct = list(0. for i in range(60))
class_total = list(0. for i in range(60))

# env vairables
os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
# os.environ["WANDB_ENTITY"] = "khurram"


def train(step_count_tb, batch_size):
    '''
        Function for training 1 epoch of network
        args:
            -step_count_tb: current epoch iteration
            -draw_graph: argument for whether to draw graph displaying number of frames in each example or not of a single batch
        returns:
            -current loss value
            -training accuracy
    '''

    class_correct = list(0. for i in range(60))
    class_total = list(0. for i in range(60))

    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    sum_curl_loss = 0
    # for batch, i in enumerate(range(0, 5 - 1, bptt)): # Size will be the number of videos in the sequence
    batch = 10
    batch_count = 0
    ntokens = len(classes)

    sum_skel_loss = []
    sum_video_loss = []
    sum_class_loss = []

    for data, targets, __ in train_loader:

        data = data.to(device)
        skel_data = data.view(75 * batch_size, 150)
        targets = targets.view(-1).to(device)
        optimizer.zero_grad()

        skel_decoded, video_decoded, pred_class = model(data)

        # calculate all the losses and perform backprop
        skel_loss = skel_criterion(skel_decoded, skel_data)
        video_loss = video_criterion(video_decoded, skel_data)

        pred_class = pred_class.view(-1, len(classes))
        class_loss = class_criterion(pred_class, targets)

        # Calculating accuracies
        _, predicted = torch.max(pred_class.view(-1, ntokens), 1)
        c = (predicted == targets).squeeze()

        for i in range(args.batch_size):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] = class_total[label] + 1

        skel_loss, video_loss, class_loss = nomalize_losses(skel_loss, video_loss, class_loss)

        sum_skel_loss.append(skel_loss.item())
        sum_video_loss.append(video_loss.item())
        sum_class_loss.append(class_loss.item())

        loss = skel_loss + video_loss + class_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        sum_curl_loss += loss.item()

        log_interval = 500
        batch += 10
        # logging interval
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                epoch, batch, len(data), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss))
            print(" Loss For each 50 batch : {}".format(total_loss))
            total_loss = 0
            step_count_tb += 1
            batch = 10
            start_time = time.time()
        batch_count += 1
        # wandb.sklearn.plot_confusion_matrix(targets.cpu().numpy().squeeze(), predicted.cpu().numpy(), labels=classes)

    print(" Mean Skel LOSS : {} and STD DEV : {}".format(np.mean(sum_skel_loss), np.std(sum_skel_loss)))
    print(" Mean Video LOSS : {} and STD DEV : {}".format(np.mean(sum_video_loss), np.std(sum_video_loss)))
    print(" Mean Class LOSS : {} and STD DEV : {}".format(np.mean(sum_class_loss), np.std(sum_class_loss)))
    print(" TOTAL LOSS : {} and SIZE DATA LOADER : {}".format(sum_curl_loss, len(train_loader)))
    train_acc = calculate_accuracy(class_correct, class_total)
    return sum_curl_loss / len(train_loader), train_acc
    # write_to_graph('train/loss', sum_curl_loss/len(train_loader), writer, step_count_tb)


def evaluate(eval_model, eval_loader, batch_size, draw_img=False, visualize=False, out_path=''):
    '''
        Function to evaluate model performance
        args:
            -eval_model: model to be used for evaluation purposes
            -eval_loader: data loader for data loading
            -draw_img: argument to draw confusion matrix at end of evaluation or not
            -visualize: argument to draw skeletons for visualization purposes
            -out_path: output path for visualization images

        returns:
            Loss and accuracy for given dataset
    '''

    class_correct = list(0. for i in range(60))
    class_total = list(0. for i in range(60))

    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    targets_ = []
    predicted_ = []
    ntokens = len(classes)

    with torch.no_grad():
        for data, targets, path in eval_loader:

            data = data.to(device)
            skel_data = data.view(75 * batch_size, 100)
            targets = targets.view(-1).to(device)
            optimizer.zero_grad()

            skel_decoded, video_decoded, pred_class = model(data)

            # calculate all the losses and perform backprop
            skel_loss = skel_criterion(skel_decoded, skel_data)
            video_loss = video_criterion(video_decoded, skel_data)
            class_loss = class_criterion(pred_class.view(-1, len(classes)), targets)

            # print(" Skeleton LOSS : {} ".format(skel_loss))
            # print(" Video LOSS : {} ".format(video_loss))
            # print(" Class LOSS : {} ".format(class_loss))

            skel_loss, video_loss, class_loss = nomalize_losses(skel_loss, video_loss, class_loss)

            # Calculating accuracies
            _, predicted = torch.max(pred_class.view(-1, ntokens), 1)
            c = (predicted == targets).squeeze()

            for i in range(args.batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1

            loss = skel_loss + video_loss + class_loss

            total_loss += loss.item()

            if visualize:  # visualize skeletons
                create_dir(out_path)
                preds = idx_class(classes, predicted.cpu().numpy())
                visualize_skeleton(path[0], preds, out_path)

        total_samples = (len(eval_loader))

    # wandb.sklearn.plot_confusion_matrix(np.array(targets_), np.array(predicted_), labels=classes)
    if draw_img:  # draw confusion_matrix
        targets_ = np.array(targets_).flatten()
        predicted_ = np.array(predicted_).flatten()
        draw_confusion_matrix(targets_, predicted_, './confusion_matrix', classes)

    val_acc = calculate_accuracy(class_correct, class_total)
    print(" TOTAL LOSS : {} and total samples : {}".format(total_loss, total_samples))
    return total_loss / total_samples, val_acc


def nomalize_losses(skel_loss, video_loss, class_loss):
    '''
    Purpose: Normalizing the losses
    and rescaling them to get better results

    :param skel_loss: skel_decoding loss
    :param video_loss: video_decoding loss
    :param class_loss: Classification loss
    :return: skel_loss, video_loss, class_loss
    '''

    # Previous configuration
    scale_value = 50
    # skel_loss = (class_loss / skel_loss) * skel_loss * (scale_value -15)
    # video_loss = (class_loss / video_loss) * video_loss * (scale_value - 10)
    # class_loss = class_loss * scale_value

    # Current configuration

    skel_loss = (class_loss / skel_loss) * skel_loss * (scale_value -15)
    video_loss = (class_loss / video_loss) * video_loss * (scale_value - 10)
    class_loss = class_loss * scale_value

    '''
    Another normalization Strategy but not very effective at the moment
    skel_loss = (skel_loss - 670739) / 98927
    video_loss = (video_loss - 664199) / 113842
    class_loss = (class_loss - 413) / 4
    skel_loss = skel_loss * (-1) if skel_loss < 0 else skel_loss
    video_loss = video_loss * (-1) if video_loss < 0 else video_loss
    class_loss = class_loss * (-1) if class_loss < 0 else class_loss
    '''
    return skel_loss, video_loss, class_loss


def calculate_accuracy(class_correct, class_total):
    acc_sum = 0
    for i in range(len(classes)):
        class_accuracy = 100 * class_correct[i] / class_total[i]

        # error_rate = 100 - class_accuracy
        acc_sum += class_accuracy
        print('%d Accuracy of %5s : %2d %% and total count : %2d ' % (i, classes[i], class_accuracy, class_total[i]))
    print('=' * 89)
    print('Mean Average Accuracy of Camera View : %2f %%' % (acc_sum / len(classes)))
    print('=' * 89)

    return acc_sum / len(classes)  # , error_rate
    # write_to_graph(split+'/Accuracy', acc_sum/60, writer, epoch)


# training arguments
parser = argparse.ArgumentParser(description="Skeleton Classification Training Script")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=100, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=100, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='./xsub_train_norm_int.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='./xsub_val_norm_float.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-hid_dim", "--nhid", type=int, default=8, help='Number of hidden dimenstions, default is 100')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--train", help="Put Model in training mode", default=True)
parser.add_argument("-f", "--frame_count", help="Frame count per video", default=60)
parser.add_argument("-bs", "--batch_shuffle", help="path to store model weights", default=False)
parser.add_argument("-c", "--checkpoint", help="path to store model weights", default="/netscratch/m_ahmed/skeleton_activity/three_network/")
parser.add_argument("-rc", "--resume_checkpoint", help="path to load model weights from for resuming training", default="./logs/output_2020-09-22 12:37:39.576905/epoch_2020-09-23 06:34:34.803320")
parser.add_argument("-r", "--resume_bool", default=False, help='Whether to resume training or start from scratch')
parser.add_argument("-ac", "--ae_checkpoint", help="path to load autoencoder from",
                    default="./autoencoder_weights/subject_video_ae_int_rgb.pth")
parser.add_argument("-pt", "--train_pickle", help="path to dir where mean and std are stored for training in pickle file",
                    default="./data/xsub_train_mean_std_f.pickle")
parser.add_argument("-pv", "--val_pickle", help="path to dir where mean and std are stored for training in pickle file",
                    default="./data/xsub_val_mean_std_f.pickle")

args = parser.parse_args()

# classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up",
#             "tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses",
#            "take off glasses", "put on a hat/cap",
#            "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping",
#            "jump up", "phone call", "play with phone/tablet",
#            "type on a keyboard", "point to something", "taking a selfie", "check time (from watch)", "rub two hands",
#            "nod head/bow", "shake head", "wipe face",
#            "salute", "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down",
#            "headache", "chest pain", "back pain", "neck pain",
#            "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing", "pat on back", "point finger", "hugging",
#            "giving object", "touch pocket", "shaking hands",
#            "walking towards", "walking apart"]

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
wandb.init(project="Skel_class", reinit=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#kwargs = {'num_workers': 0, 'pin_memory': True} if device == 'cuda' else {}

if args.train:

    train_dataset = SkeletonsDataset(args.train_data, args.batch_size, args.train_pickle, args.val_pickle)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.batch_shuffle)
    for data, targets, __ in train_loader:
        print('data shape: ', data.shape)
        print(targets.shape)
    # eval_dataset = SkeletonsDataset(args.eval_data, args.eval_batch_size, args.train_pickle, args.val_pickle)
    # eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=args.batch_shuffle, **kwargs)

    # '''
    #     Loading all the models here 
    #     Classification Network     
    #     Defining the criterion for losses
    # '''
    # model = classification_network_128(num_feature=128, num_class=len(classes), batch_size=args.batch_size).to(device)

    # skel_criterion = nn.MSELoss(reduction='sum')
    # video_criterion = nn.MSELoss(reduction='sum')
    # class_criterion = nn.CrossEntropyLoss(reduction='sum')

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # print(model)


    # best_val_loss = float("inf")
    # max_epochs = args.epochs  # number of epochs
    # step_count_tb = 1  # xaxis for calculating loss value for tensorboard

    # # Saving and writing model as a state_dict
    # # Training procedure starts
    # current_date_time = str(datetime.datetime.now()).split(",")[0]

    # output_path = args.checkpoint + "output_" + str(current_date_time)
    # create_dir(output_path)  # creating the directory where epochs will be saved
    # skeleton_output = os.path.join(output_path, 'skeleton_diagrams')

    # output_log = []
    # '''
    # Resuming from the specific check point
    # '''
    # if args.resume_bool:
    #     # Resuming the model from the specific checkpoint
    #     checkpoint = torch.load(args.resume_checkpoint)
    #     model.load_state_dict(checkpoint['model_state_dict'])

    # lr_change_count = 0
    # epoch_output_path = output_path + "/epoch_" + str(current_date_time)
    # for epoch in range(0, max_epochs):

    #     epoch_start_time = time.time()
    #     current_date_time = str(datetime.datetime.now()).split(",")[0]


    #     tr_loss, tr_acc = train(step_count_tb, args.batch_size)

    #     print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.4f} | '
    #           ' Train Accuracy '.format(epoch, (time.time() - epoch_start_time),
    #                                      tr_loss, tr_acc))

    #     if epoch == max_epochs - 1:
    #         print('***** in this line part *****')
    #         val_loss, val_acc = evaluate(model, eval_loader, args.batch_size, draw_img=False, visualize=False,
    #                                      out_path=skeleton_output)
    #     else:
    #         val_loss, val_acc = evaluate(model, eval_loader, args.batch_size)

    #     output_log.append(
    #         'train_loss: ' + str(tr_loss) + '\t' + 'training_acc: ' + str(tr_acc) + '\t' + 'val_loss: ' + str(
    #             val_loss) + '\t' + 'val_acc: ' + str(val_acc) + '\n')
    #     wandb.log({"train_loss": tr_loss, "training_acc": tr_acc, "test_loss": val_loss, "test_acc": val_acc,
    #                "learing_rate": optimizer.param_groups[0]['lr']})
    #     print('-' * 89)
    #     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),
    #                                      val_loss))

    #     output_log.append('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time) ,val_loss))
    #     output_log.append('\n')
    #     print('-' * 89)
    #     # write_to_graph('Val/loss', val_loss, writer, epoch)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_model = model
    #         best_epoch = epoch
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': best_model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': best_val_loss
    #         }, epoch_output_path)


    #     if epoch % 100 == 0 and lr_change_count < 3: #reducing learning rate for experiment
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = param_group["lr"] * (0.993 ** epoch)
    #         lr_change_count +=1

    # with open(output_path + "_" + "log.txt", "w") as outfile:
    #     outfile.write("\n".join(output_log))



    # for data, targets, __ in train_loader:
    #     targets = targets.view(-1).to(device)
    #
    #     for i in range(args.batch_size):
    #         label = targets[i]
    #         class_total[label] = class_total[label] + 1
    # print(class_total)
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

