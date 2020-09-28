# Standard Library
import os
import wandb
import logging
import argparse
from statistics import mean
from torch.nn import CrossEntropyLoss, MSELoss
from model import AutoEncoderWithClassifier
from data_source_reader_video import *

# Local Modules


#os.chdir("../")
#from utils import create_dir

os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def evaluate(eval_model, eval_loader, reshape_size):
    
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

    eval_model.eval() # Turn on the evaluation mode
    criterion1 = MSELoss()
    criterion2 = CrossEntropyLoss()
    with torch.no_grad():
        losses = []
        for data, targets, __, in eval_loader:

            data = data.view(-1, reshape_size)

            # Forward pass only
            data = data.to(device)
            targets = targets.view(-1).to(device)

            decoded_output, class_output = model(data)

            # Calculating accuracies
            _, predicted = torch.max(class_output.view(-1, len(classes)), 1)
            c = (predicted == targets).squeeze()

            for i in range(args.batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1

            # Calculating accuracies code ends here

            loss1 = criterion1(decoded_output, data)
            loss2 = criterion2(class_output.view(-1, len(classes)), targets)
            loss = loss1 + loss2

            losses.append(loss.item())
        test_acc= calculate_accuracy(class_correct, class_total)
        wandb.log({"Test Accuracy": test_acc})

        return mean(losses)

def train_model(model, train_loader, eval_loader, lr, epochs, logging, out_path, reshape_size):

    '''
        Function to train model
        args:
            -model: model to be used for training purposes
            -train_loader: data loader for data loading
            -eval_loader: to be passed to the evaluate function after every third epoch
            -lr : learning rate, reduced after every third epoch
            -epochs: maximum epoch parse from argparese
            -logging : True then write to log files
            -out_path: location where model needs to be saved
            -reshape_size: value based on skeleton or video level processing

        returns:
            model : trained model that can be evaluated and saved for later use.
    '''



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('***** Training on '+str(device)+' *****')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    criterion1 = MSELoss()
    criterion2 = CrossEntropyLoss()

    print_once=True
    curr_loss = float("inf")
    lr_change_count = 0
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        count_batch = 0

        class_correct = list(0. for i in range(60))
        class_total = list(0. for i in range(60))

        logging.info('*******Starting Training for epoch {} *******'.format(epoch))
        # iterating over the dataset to create a whole skeleton sequence
        for data, targets, __, in train_loader:

            data = data.view(-1, reshape_size)

            if count_batch % 20000 == 0:
                logging.info('{} data done'.format(count_batch))

            #for seq_true in dataset:
            #seq_true = torch.tensor(seq_true.numpy()[np.newaxis, :], dtype=torch.float)
            # model calculating both losses together
            data = data.to(device)
            targets = targets.view(-1).to(device)

            optimizer.zero_grad()

            decoded_output, class_output = model(data)

            # Calculating accuracies
            _, predicted = torch.max(class_output.view(-1, len(classes)), 1)
            c = (predicted == targets).squeeze()

            for i in range(args.batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1

            # Calculating accuracies code ends here

            loss1 = criterion1(decoded_output, data)
            loss2 = criterion2(class_output.view(-1, len(classes)), targets)
            loss = loss1 + loss2
            # Backward pass

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            count_batch+=1

            '''
            Calculating Training Accuracy here
            saving in variable here, maybe use later to log in Wandb
            '''
        train_acc = calculate_accuracy(class_correct, class_total)

        logging.info('Done with epoch {}'.format(epoch))
                
        if epoch % 5 == 0:
            val_loss = evaluate(model, eval_loader, reshape_size)
            logging.info('Val Loss {}'.format(val_loss))
            wandb.log({"Test loss": val_loss})
            if val_loss < curr_loss:
                curr_loss = val_loss
                logging.info('Saving model')
                torch.save(model.state_dict(), os.path.join(out_path, 'subject_video_ae_class_int_rgb.pth'))

            if lr_change_count < 3:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * (0.993 ** epoch)
                lr_change_count +=1

        logging.info("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
        print("Epoch: {}, Loss: {} ".format(str(epoch), str(mean(losses))))
        wandb.log({"train_loss": mean(losses), "learing_rate": optimizer.param_groups[0]['lr']})
        wandb.log({"Train Accuracy": train_acc})
            
    return model

def calculate_accuracy(class_correct, class_total):
    acc_sum = 0
    for i in range(len(classes)):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        # error_rate = 100 - class_accuracy
        acc_sum += class_accuracy
        print('%d Accuracy of %5s : %2d %% and total count : %2d ' % (i, classes[i], class_accuracy, class_total[i]))
    print('=' * 89)
    print('Mean Average Accuracy of Camera View : %2f %%' % (acc_sum / 60))
    print('=' * 89)

    return acc_sum / len(classes)  # , error_rate
    # write_to_graph(split+'/Accuracy', acc_sum/60, writer, epoch)

parser = argparse.ArgumentParser(description="Skeleton Autoencoders training ")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=100, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=100, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='../xsub_train_norm_rgb.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='../xsub_val_norm_rgb.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-o", "--output", help='Path to save autoencoder weights', default='../data/data_rgb_new/ae_class_weights/')
parser.add_argument("-e", "--epochs", type=int, default=200, help='Number of epochs to train model for')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--tr_type", default='video', help='Train on video or skeleton')
parser.add_argument("-r", "--resume_bool", default=False, help='Train on video or skeleton')
parser.add_argument("-cp", "--check_point", help="path to load autoencoder from", default="autoencoder_weights/subject_skeleton_autoencoder_int_rgb.pth")


args = parser.parse_args()
# initalize wandb
wandb.init(project="Auto Encoders", reinit=True)

LOG_FILENAME = 'experiment_real_Skeleton_RGB.log'
# Set up a specific logger with our desired output level
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.INFO)
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

train_dataset = SkeletonsDataset(args.train_data, args.batch_size, '../data/data_rgb_new/xsub_train_mean_std.pickle', '../data/data_rgb_new/xsub_val_mean_std.pickle')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

eval_dataset = SkeletonsDataset(args.eval_data, args.eval_batch_size, '../data/data_rgb_new/xsub_train_mean_std.pickle', '../data/data_rgb_new/xsub_val_mean_std.pickle')
eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
if not os.path.exists(args.output):
    os.mkdir(args.output)

model = AutoEncoderWithClassifier().cuda()

if args.resume_bool:
    # Resuming the model from the specific checkpoint
    model = AutoEncoderWithClassifier()
    model.load_state_dict(torch.load(args.check_point))

#RAE(seq_len, num_features, 75)
if args.tr_type == 'video':
    model = train_model(model, train_loader, eval_loader, args.learning_rate, args.epochs, logging, args.output, 7500)
else:
    model = train_model(model, train_loader, eval_loader, args.learning_rate, args.epochs, logging, args.output, 100)



#torch.save(model.state_dict(), './sim_autoencoder.pth')
# encoder, decoder, embeddings, f_loss = QuickEncode(
#     sequences,
#     embedding_dim=2,
#     logging=True
# )

#
# test_encoding = encoder(torch.tensor([[1.0], [4.0], [2.0], [3.0]]))
# test_decoding = decoder(test_encoding)
#
# print()
# print(test_encoding)
# print(test_decoding)