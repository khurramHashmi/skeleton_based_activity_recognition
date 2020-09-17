# Standard Library
from statistics import mean
import os
import wandb
#os.chdir("./../")
from data_source_reader import *
# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
import argparse
import sys
# Local Modules
# os.chdir("./embeddings/")
from model import RAE, simple_autoencoder
os.chdir("../")

# os.environ["WANDB_MODE"] = "dryrun"
os.environ["WANDB_API_KEY"] = "cbf5ed4387d24dbdda68d6de22de808c592f792e"
os.environ["WANDB_ENTITY"] = "khurram"
# initalize wandb

def prepare_dataset(sequences):
    if type(sequences) == list:
        dataset = []
        for sequence in sequences:
            updated_seq = []
            for vec in sequence:
                if type(vec) == list:
                    updated_seq.append([float(elem) for elem in vec])
                else: # Sequence is 1-D
                    updated_seq.append([float(vec)])

            dataset.append(torch.tensor(updated_seq))
    elif type(sequences) == torch.tensor:
        dataset = [sequences[i] for i in range(len(sequences))]
    # dataset = sequences
    shape = torch.stack(dataset).shape
    assert(len(shape) == 3)

    return dataset, shape[1], shape[2]


def train_model(model, train_loader, lr, epochs, logging):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # criterion = CrossEntropyLoss()
    criterion = MSELoss()
    print_once=True
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        print('*******Starting Training for epoch {} *******'.format(epoch))
        # iterating over the dataset to create a whole skeleton sequence
        for data, __, __ in train_loader:
            
            seq_true = data.view(-1, 6000)
            
            if print_once:
                print('Example shape: ',seq_true.shape)
                print_once=False

            # Reduces learning rate every 3 epochs
            if not epoch % 3:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * (0.993 ** epoch)

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

        if logging:
            print("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
            wandb.log({"train_loss": mean(losses), "learing_rate": optimizer.param_groups[0]['lr']})


#########
# EXPORTS
#########
#
#
# def QuickEncode(sequences, embedding_dim, logging=False, lr=0.01, epochs=500):

    # dataset, seq_len, num_features = prepare_dataset(sequences)



# if __name__ == "__main__":

parser = argparse.ArgumentParser(description="Skeleton Autoencoders training ")
parser.add_argument("-lr", "--learning_rate", default=5.0, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size", default=100, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size", default=10, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='xsub_val_rgb.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='xsub_val_rgb.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data", help='Path to test data')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')

args = parser.parse_args()

wandb.init(project="Auto Encoders", reinit=True)

train_dataset = SkeletonsDataset(args.train_data, args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# eval_dataset = SkeletonsDataset(args.eval_data, args.eval_batch_size)
# eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)


# sequences = [[1, 4, 12, 13], [9, 6, 2, 1], [3, 3, 14, 11]]
# print(sequences)

#
# dataset = sequences
seq_len =1
num_features = 100

model = simple_autoencoder().cuda()
#RAE(seq_len, num_features, 75)
embeddings, f_loss = train_model(model, train_loader, 0.001, args.epochs, True)
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