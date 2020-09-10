from data_source_reader import SkeletonsDataset
from model_transformer import *
import torch
import torch.nn.functional as F
import time
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

writer = SummaryWriter()
class_correct = list(0. for i in range(60))
class_total = list(0. for i in range(60))

def train(step_count_tb):

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    # for batch, i in enumerate(range(0, 5 - 1, bptt)): # Size will be the number of videos in the sequence
    batch=5
    for data, targets in train_loader:

        data = data.to(device)
        targets = targets.view(-1).to(device)

        optimizer.zero_grad()
        output = model(data)

        # loss = criterion(output.view(-1, ntokens), targets)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        #To Calculate classification
        _, predicted = torch.max(output.view(-1, ntokens), 1)
        c = (predicted == targets).squeeze()
        for i in range(eval_batch_size):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


        total_loss += loss.item()
        log_interval = 200
        batch += 5
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            write_to_graph('train/loss',cur_loss,writer,step_count_tb)
            total_loss = 0
            step_count_tb+=1
            batch = 5
            start_time = time.time()
    calculate_accuracy()
def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for data, targets in eval_loader:
            # giving the tensors to Cuda
            data = data.to(device)
            targets = targets.view(-1).to(device) # Linearize the target tensor to match the shape
            output = eval_model(data)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
        total_samples = (len(eval_loader) * eval_batch_size)
    return total_loss / total_samples

def classify_evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for data, targets in eval_loader:
            # giving the tensors to Cuda
            data = data.to(device)
            targets = targets.view(-1).to(device) # Linearize the target tensor to match the shape
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            _, predicted = torch.max(output_flat, 1)
            c = (predicted == targets).squeeze()
            for i in range(eval_batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    calculate_accuracy()
    return total_loss / (len(eval_loader)*eval_batch_size)

def calculate_accuracy():
    for i in range(60):
        print('%d Accuracy of %5s : %2d %%' % (i, classes[i], 100 * class_correct[i] / class_total[i]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up", "clapping", "reading", "writing"
           ,"tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses", "take off glasses", "put on a hat/cap",
           "take off a hat/cap", "cheer up","hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "phone call", "play with phone/tablet",
           "type on a keyboard", "point to something","taking a selfie", "check time (from watch)", "rub two hands", "nod head/bow", "shake head", "wipe face",
           "salute", "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down", "headache", "chest pain", "back pain", "neck pain",
           "nausea/vomiting", "fan self", "punch/slap",	"kicking", "pushing", "pat on back", "point finger", "hugging", "giving object", "touch pocket", "shaking hands",
           "walking towards", "walking apart"]

train_batch_size = 5

train_dataset = SkeletonsDataset('data/train.tsv')
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, **kwargs)
#
eval_batch_size = 5
eval_dataset = SkeletonsDataset('data/old_train_with180_samples.tsv')
eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, **kwargs)

# Defining Model with parameters
ntokens = len(classes) # the size of vocabulary #change number of tokens from 15400 to 154
emsize = 100 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multi head attention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)



best_val_loss = float("inf")
max_epochs = 10 # number of epochs
step_count_tb = 1 # xaxis for calculating loss value for tensorboard



# Saving and writing model as a state_dict
# Training procedure starts
time_check = time.time()
output_path = "./logs/output_"+str(time_check)
create_dir(output_path) # creating the directory where epochs will be saved
for epoch in range(1,  max_epochs):
    epoch_start_time = time.time()
    epoch_output_path = output_path +"/epoch_"+str(epoch_start_time)
    train(step_count_tb)
    val_loss = evaluate(model)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    write_to_graph('Val/loss', val_loss, writer, epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss
        }, output_path)
    scheduler.step()

# loading the model again to evaluate for the test set.
# checkpoint = torch.load("output/out_model_camera_view_train_1599686144.9230924")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# test_loss = classify_evaluate(model)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)



# max = 300
# sk = SkeletonsDataset('data/train.tsv')
# for i in range(sk.__len__()):
#     x , _ = sk.__getitem__(i)
#     if len(x) > max:
#         print(len(x))
# for data,target in train_loader:
#     # x1 = (len(data[0]))
#     if len(data[0]) > max:
#         print(str(len(data[0])))
#         max = len(data)
# print(str(max))