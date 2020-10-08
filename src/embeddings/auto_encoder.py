# Standard Library
import os
import wandb
import torch
import logging
import argparse
from statistics import mean
from torch.nn import CrossEntropyLoss, MSELoss
# Local Modules
#os.chdir("./../")
from data_source_reader_video import *
from model import SimpleAutoEncoderVideo
#os.chdir("../")
#from utils import create_dir

os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"



def evaluate(eval_model, eval_loader, reshape_size):
    
    '''
        Function to evaluate model performance
        args:
            -eval_model: model to be used for evaluation purposes
            -eval_loader: data loader for data loading 

        returns:
            Loss and accuracy for given dataset
    '''

    eval_model.eval() # Turn on the evaluation mode
    criterion = MSELoss()
    with torch.no_grad():
        losses = []
        for data, __, __, in eval_loader:

            seq_true = data.view(-1, reshape_size)

            # Forward pass
            seq_true = seq_true.cuda()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)

            # Backward pass
            losses.append(loss.item())
        
        return np.mean(losses)

def train_model(model, train_loader, eval_loader, lr, epochs, logging, out_path, reshape_size):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('***** Training on '+str(device)+' *****')
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # criterion = CrossEntropyLoss()
    criterion = MSELoss()
    print_once=True
    curr_loss = float("inf")
    lr_change_count = 0
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        count_batch = 0
        logging.info('*******Starting Training for epoch {} *******'.format(epoch))
        # iterating over the dataset to create a whole skeleton sequence
        for data, __, __, in train_loader:
            
            seq_true = data.view(-1, reshape_size)
            
            if print_once:
                logging.info('Example shape: {}'.format(seq_true.shape))
                print_once=False
            
            if count_batch % 20000 == 0:
                logging.info('{} data done'.format(count_batch))
            
            #for seq_true in dataset:
            #seq_true = torch.tensor(seq_true.numpy()[np.newaxis, :], dtype=torch.float)
            # Forward pass
            seq_true = seq_true.cuda()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)

            # Backward pass
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            count_batch += 1
            if count_batch % 200 == 0:
                logging.info('Done with batch {}'.format(count_batch))

        logging.info('Done with epoch {}'.format(epoch))
                
        if epoch % 3 == 0:
            val_loss = evaluate(model, eval_loader, reshape_size)
            logging.info('Val Loss {}'.format(val_loss))
            wandb.log({"val_loss": val_loss})
            if val_loss < curr_loss:
                curr_loss = val_loss
                logging.info('Saving model')
                torch.save(model.state_dict(), os.path.join(out_path, 'subject_video_autoencoder_int_rgb_512.pth'))
            wandb.log({"test_loss": val_loss})
            if lr_change_count < 3:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * (0.993 ** epoch)
                lr_change_count +=1

        logging.info("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
        print("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
        wandb.log({"train_loss": mean(losses), "learing_rate": optimizer.param_groups[0]['lr']})

            
    return model

parser = argparse.ArgumentParser(description="Skeleton Autoencoders training ")
parser.add_argument("-lr", "--learning_rate", default=5.0, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=100, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=100, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='../xsub_train_norm_rgb.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='../xsub_val_norm_rgb.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-o", "--output", help='Path to save autoencoder weights', default='../data/data_rgb_new/autoencoder_weights_video/')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--tr_type", default='video', help='Train on video or skeleton')
parser.add_argument("-r", "--resume_bool", default=False, help='Train on video or skeleton')
parser.add_argument("-rc", "--resume_checkpoint", help="path to store model weights", default="../data/data_rgb_new/autoencoder_weights_video/subject_video_autoencoder_int_rgb_512.pth")


args = parser.parse_args()
# initalize wandb
wandb.init(project="Auto Encoders", reinit=True)

LOG_FILENAME = 'experiment_video_AE_only_RGB.log'
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

model = SimpleAutoEncoderVideo().cuda()

if args.resume_bool:
    # Resuming the model from the specific checkpoint
    model = SimpleAutoEncoderVideo()
    model.load_state_dict(torch.load(args.resume_checkpoint))

#RAE(seq_len, num_features, 75)
if args.tr_type == 'video':
    model = train_model(model, train_loader, eval_loader, 0.001, args.epochs, logging, args.output, 7500)
else:
    model = train_model(model, train_loader, eval_loader, 0.001, args.epochs, logging, args.output, 100)



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