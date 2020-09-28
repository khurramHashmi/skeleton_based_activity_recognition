import time
import datetime
import wandb
import argparse
from utils import *
from model_transformer import *
from data_source_reader_video import SkeletonsDataset
from torch.utils.data import DataLoader
from embeddings.model import SimpleAutoEncoderVideo
from embeddings.model import classification_network

class_correct = list(0. for i in range(60))
class_total = list(0. for i in range(60))

# env vairables
# os.environ["WANDB_MODE"] = "dryrun"
os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"


def train(step_count_tb, reshape_size, draw_graph=False, graph_out=''):
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
    for data, targets, __ in train_loader:

        # Calculate embeddings
        embeddings = extract_embeddings(data, reshape_size)
        data = embeddings.to(device)
        targets = targets.view(-1).to(device)
        optimizer.zero_grad()
        output = model(data)

        # Calculating accuracies
        _, predicted = torch.max(output.view(-1, ntokens), 1)
        c = (predicted == targets).squeeze()


        for i in range(args.batch_size):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] = class_total[label] + 1

        # calculate loss and perform backprop
        loss = criterion(output.view(-1, ntokens), targets)
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
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            print(" Loss For each 50 batch : {}".format(total_loss))
            total_loss = 0
            step_count_tb += 1
            batch = 10
            start_time = time.time()
        batch_count += 1
        # wandb.sklearn.plot_confusion_matrix(targets.cpu().numpy().squeeze(), predicted.cpu().numpy(), labels=classes)

    train_acc = calculate_accuracy(class_correct, class_total)
    return sum_curl_loss / len(train_loader), train_acc
    # write_to_graph('train/loss', sum_curl_loss/len(train_loader), writer, step_count_tb)


def extract_embeddings(example, reshape_size):
    # preprocessing data
    example = example.view(-1, reshape_size).to(device)
    # forward pass to get embeddings
    output = ae_model.encoder(example)

    return output


def evaluate(eval_model, eval_loader, reshape_size, draw_img=False, visualize=False, out_path=''):
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
            # giving the tensors to Cuda
            embeddings = extract_embeddings(data, reshape_size)
            data = embeddings.to(device)
            targets_.append(targets.numpy().squeeze())
            targets = targets.view(-1).to(device)  # Linearize the target tensor to match the shape
            output = eval_model(data)

            # Calculating accuracies
            _, predicted = torch.max(output.view(-1, ntokens), 1)
            predicted_.append(predicted.cpu().numpy())
            c = (predicted == targets).squeeze()
            for i in range(args.batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1

            total_loss += len(data) * criterion(output.view(-1, len(classes)), targets).item()

            if visualize:  # visualize skeletons
                create_dir(out_path)
                preds = idx_class(classes, predicted.cpu().numpy())
                visualize_skeleton(path[0], preds, out_path)

        total_samples = (len(eval_loader) * args.eval_batch_size)

    # wandb.sklearn.plot_confusion_matrix(np.array(targets_), np.array(predicted_), labels=classes)
    if draw_img:  # draw confusion_matrix
        targets_ = np.array(targets_).flatten()
        predicted_ = np.array(predicted_).flatten()
        draw_confusion_matrix(targets_, predicted_, './confusion_matrix', classes)

    val_acc = calculate_accuracy(class_correct, class_total)
    return total_loss / total_samples, val_acc


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
parser.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=50, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='./xsub_val_norm_rgb.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='./xsub_val_norm_rgb.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-hid_dim", "--nhid", type=int, default=8, help='Number of hidden dimenstions, default is 100')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--train", help="Put Model in training mode", default=True)
parser.add_argument("-f", "--frame_count", help="Frame count per video", default=60)
parser.add_argument("-c", "--checkpoint", help="path to store model weights", default="./logs/")
parser.add_argument("-rc", "--resume_checkpoint", help="path to store model weights", default="./logs/output_1600710572.0355484/epoch_1600748333.0358984")
parser.add_argument("-r", "--resume_bool", default=True, help='Train on video or skeleton')
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
wandb.init(project="Skel_class", reinit=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

if args.train:

    # train_dataset = SkeletonsDataset(args.train_data, args.batch_size, None,None)

    train_dataset = SkeletonsDataset(args.train_data, args.batch_size, './data/data_rgb_new/xsub_val_mean_std.pickle', './data/data_rgb_new/xsub_val_mean_std.pickle')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print("length of train data : {}".format(train_dataset.__len__()))
    # eval_dataset = SkeletonsDataset(args.eval_data, args.eval_batch_size, None,None)
    eval_dataset = SkeletonsDataset(args.train_data, args.batch_size, './data/data_rgb_new/xsub_val_mean_std.pickle', './data/data_rgb_new/xsub_val_mean_std.pickle') #only val is using for now
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)
    print("length of train data : {}".format(eval_dataset.__len__()))


    '''
        Loading multi-class classification model
    '''
    model = classification_network(num_feature=128, num_class=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(model)


    # load autoencoder
    ae_model = SimpleAutoEncoderVideo()
    ae_model.load_state_dict(torch.load(args.ae_checkpoint))
    ae_model.eval().to(device)

    best_val_loss = float("inf")
    max_epochs = args.epochs  # number of epochs
    step_count_tb = 1  # xaxis for calculating loss value for tensorboard

    # Saving and writing model as a state_dict
    # Training procedure starts
    current_date_time = str(datetime.datetime.now()).split(",")[0]

    output_path = args.checkpoint + "output_" + str(current_date_time)
    create_dir(output_path)  # creating the directory where epochs will be saved
    graph_out = os.path.join(output_path, 'graph')
    create_dir(graph_out)
    skeleton_output = os.path.join(output_path, 'skeleton_diagrams')
    print('Saving graph in dir: ', graph_out)

    output_log = []
    draw_graph = True

    '''
    Resuming from the specific check point
    '''
    if args.resume_bool:
        # Resuming the model from the specific checkpoint
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.on_load_checkpoint(checkpoint)

    for epoch in range(0, max_epochs):

        epoch_start_time = time.time()
        current_date_time = str(datetime.datetime.now()).split(",")[0]

        epoch_output_path = output_path + "/epoch_" + str(current_date_time)
        tr_loss, tr_acc = train(step_count_tb, 7500, draw_graph, graph_out)
        draw_graph = False
        print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.4f} | '
              'Train ppl {:8.2f} | Train Accuracy '.format(epoch, (time.time() - epoch_start_time),
                                         tr_loss, math.exp(tr_loss), tr_acc))

        if epoch == max_epochs - 1:
            print('***** in this line part *****')
            val_loss, val_acc = evaluate(model, eval_loader, 7500, draw_img=False, visualize=False,
                                         out_path=skeleton_output)
        else:
            val_loss, val_acc = evaluate(model, eval_loader, 7500)

        output_log.append(
            'train_loss: ' + str(tr_loss) + '\t' + 'training_acc: ' + str(tr_acc) + '\t' + 'val_loss: ' + str(
                val_loss) + '\t' + 'val_acc: ' + str(val_acc) + '\n')
        wandb.log({"train_loss": tr_loss, "training_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc,
                   "learing_rate": optimizer.param_groups[0]['lr']})
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))

        output_log.append('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
                          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                     val_loss, math.exp(val_loss)))
        output_log.append('\n')
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

    with open(output_path + "_" + "log.txt", "w") as outfile:
        outfile.write("\n".join(output_log))

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

