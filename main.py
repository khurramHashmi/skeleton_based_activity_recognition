from data_source_reader import SkeletonsDataset
from model_transformer import *
import torch
import time
from torch.utils.data import DataLoader, Dataset

def train():

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    # for batch, i in enumerate(range(0, 5 - 1, bptt)): # Size will be the number of videos in the sequence
    batch=20

    for data, targets in train_loader:

        data = data.to(device)
        targets = targets.view(-1).to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

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
            total_loss = 0
            batch = 20
            start_time = time.time()


def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.

    # ntokens = 15400
    with torch.no_grad():
        for data, targets in eval_loader:
            # giving the tensors to Cuda
            data = data.to(device)
            targets = targets.view(-1).to(device) # Linearize the target tensor to match the shape
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(targets) - 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# Defining Model with parameters
ntokens = 40860 # the size of vocabulary #change number of tokens from 15400 to 154
emsize = 100 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

train_batch_size = 1
train_dataset = SkeletonsDataset('data/train.tsv')
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, **kwargs)

eval_batch_size = 1
eval_dataset = SkeletonsDataset('data/val.tsv')
eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, **kwargs)

best_val_loss = float("inf")
max_epochs = 100 # number of epochs

# print(str(len(train_dataset)))
# print(str(len(eval_dataset)))
#
# print(str(len(train_loader)))
# print(str(len(eval_loader)))
#
# for data, target in train_loader:
#     y = (target.shape)
#     x = (data.shape)
#     a,b = (data[0],target[0])


for epoch in range(1,  max_epochs):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


test_loss = evaluate(best_model)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)