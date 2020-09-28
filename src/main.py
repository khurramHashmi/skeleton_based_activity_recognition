import time
import wandb
import argparse
from utils import *
from model_transformer import *
from data_source_reader_video import SkeletonsDataset
from torch.utils.data import DataLoader
from embeddings.model import simple_autoencoder

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

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    sum_curl_loss = 0
    # for batch, i in enumerate(range(0, 5 - 1, bptt)): # Size will be the number of videos in the sequence
    batch=10
    batch_count=0    
    for data, targets, __, frame_count in train_loader:
        

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
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            step_count_tb+=1
            batch = 10
            start_time = time.time()
        batch_count += 1
        # wandb.sklearn.plot_confusion_matrix(targets.cpu().numpy().squeeze(), predicted.cpu().numpy(), labels=classes)

        if draw_graph: # whether to draw graph for frame distribution or not
            activity = idx_class(classes, targets.cpu().numpy())
            barplot('activity','frame_count', pd.DataFrame({'activity':activity, 'frame_count':frame_count.numpy()}), 'batch_{}'.format(batch_count), graph_out)

    train_acc = calculate_accuracy(class_correct, class_total)
    return sum_curl_loss/len(train_loader), train_acc
    #write_to_graph('train/loss', sum_curl_loss/len(train_loader), writer, step_count_tb)

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

    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    targets_ = []
    predicted_ = []

    with torch.no_grad():
        for data, targets, path, __ in eval_loader:
            # giving the tensors to Cuda
            embeddings = extract_embeddings(data, reshape_size)
            data = embeddings.to(device)
            targets_.append(targets.numpy().squeeze())
            targets = targets.view(-1).to(device) # Linearize the target tensor to match the shape
            output = eval_model(data)

            # Calculating accuracies
            _, predicted = torch.max(output.view(-1, ntokens), 1)
            predicted_.append(predicted.cpu().numpy())
            c = (predicted == targets).squeeze()
            for i in range(args.batch_size):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] = class_total[label] + 1
            
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()

            if visualize: # visualize skeletons
                create_dir(out_path)
                preds = idx_class(classes, predicted.cpu().numpy())
                visualize_skeleton(path[0], preds, out_path)

        total_samples = (len(eval_loader) * args.eval_batch_size)
    
    # wandb.sklearn.plot_confusion_matrix(np.array(targets_), np.array(predicted_), labels=classes)    
    if draw_img: # draw confusion_matrix
        targets_ = np.array(targets_).flatten()
        predicted_ = np.array(predicted_).flatten()
        draw_confusion_matrix(targets_, predicted_, './confusion_matrix', classes)

    val_acc = calculate_accuracy(class_correct, class_total)
    return total_loss / total_samples, val_acc

def calculate_accuracy(class_correct, class_total):

    acc_sum = 0
    for i in range(len(classes)):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        #error_rate = 100 - class_accuracy
        acc_sum +=class_accuracy
        print('%d Accuracy of %5s : %2d %%' % (i, classes[i], class_accuracy))
    print('=' * 89)
    print('Mean Average Accuracy of Camera View : %2f %%' % (acc_sum/60))
    print('=' * 89)

    return acc_sum/len(classes)


# training arguments
parser = argparse.ArgumentParser(description="Skeleton Classification Training Script")
parser.add_argument("-lr", "--learning_rate", default=5.0, type=float, help="Learning rate of model. Default 5.0")
parser.add_argument("-b", "--batch_size",  default=5, type=int, help="Batch Size for training")
parser.add_argument("-eb", "--eval_batch_size",  default=5, type=int, help="Batch Size for evaluation")
parser.add_argument("-tr_d", "--train_data", default='./xsub_val_rgb.csv', help='Path to training data')
parser.add_argument("-ev_d", "--eval_data", default='./xsub_val_rgb.csv', help='Path to eval data')
parser.add_argument("-ts_d", "--test_data",  help='Path to test data')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs to train model for')
parser.add_argument("-hid_dim", "--nhid", type=int, default=8, help='Number of hidden dimenstions, default is 100')
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help='Dropout value, default is 0.2')
parser.add_argument("-t", "--train", help="Put Model in training mode", default=True)
parser.add_argument("-f", "--frame_count", help="Frame count per video", default=60)
parser.add_argument("-c", "--checkpoint", help="path to store model weights", default="./logs/")
parser.add_argument("-ac", "--ae_checkpoint", help="path to load autoencoder from", default="autoencoder_weights/subject_skeleton_autoencoder_int_rgb.pth")

args = parser.parse_args()

classes = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up", "clapping", "reading", "writing"
           ,"tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses", "take off glasses", "put on a hat/cap",
           "take off a hat/cap", "cheer up","hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "phone call", "play with phone/tablet",
           "type on a keyboard", "point to something","taking a selfie", "check time (from watch)", "rub two hands", "nod head/bow", "shake head", "wipe face",
           "salute", "put palms together", "cross hands in front", "sneeze/cough", "staggering", "falling down", "headache", "chest pain", "back pain", "neck pain",
           "nausea/vomiting", "fan self", "punch/slap",	"kicking", "pushing", "pat on back", "point finger", "hugging", "giving object", "touch pocket", "shaking hands",
           "walking towards", "walking apart"]

# initalize wandb 
wandb.init(project="Skeleton Classification", reinit=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

if args.train:

    train_dataset = SkeletonsDataset(args.train_data,args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    eval_dataset = SkeletonsDataset(args.eval_data,args.eval_batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)

    # Defining Model with parameters
    ntokens = len(classes) # the size of vocabulary #change number of tokens from 15400 to 154
    emsize = 8 # embedding dimension
    nhid = args.nhid # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multi head attention models
    dropout = args.dropout # the dropout value
    # create transformer model
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, args.batch_size, args.frame_count, dropout).to(device)

    # load autoencoder
    ae_model = simple_autoencoder()
    ae_model.load_state_dict(torch.load(args.ae_checkpoint))
    ae_model.eval().to(device)
    # ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    # scheduler.get

    best_val_loss = float("inf")
    max_epochs = args.epochs # number of epochs
    step_count_tb = 1 # xaxis for calculating loss value for tensorboard

    # Saving and writing model as a state_dict
    # Training procedure starts
    time_check = time.time()
    output_path = args.checkpoint + "output_"+str(time_check)
    create_dir(output_path) # creating the directory where epochs will be saved
    graph_out=os.path.join(output_path,'graph')
    create_dir(graph_out)
    skeleton_output = os.path.join(output_path, 'skeleton_diagrams')
    print('Saving graph in dir: ', graph_out)

    output_log = []

    draw_graph=True
    for epoch in range(0,  max_epochs):
        epoch_start_time = time.time()
        epoch_output_path = output_path +"/epoch_"+str(epoch_start_time)
        tr_loss, tr_acc = train(step_count_tb,100, draw_graph, graph_out)
        draw_graph=False
        print('| end of epoch {:3d} | time: {:5.2f}s | Train loss {:5.4f} | '
            'Train ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        tr_loss, math.exp(tr_loss)))

        if epoch == max_epochs-1:
            print('***** in this line part *****')
            val_loss, val_acc = evaluate(model, eval_loader, 100, draw_img=False, visualize=False, out_path=skeleton_output)
        else:
            val_loss, val_acc = evaluate(model, eval_loader, 100)
        
        output_log.append('train_loss: '+str(tr_loss)+'\t'+ 'training_acc: '+str(tr_acc) + '\t' + 'val_loss: ' +str(val_loss) + '\t' + 'val_acc: '+str(val_acc) + '\n')    
        wandb.log({"train_loss":tr_loss, "training_acc":tr_acc, "val_loss":val_loss, "val_acc":val_acc, "learing_rate":optimizer.param_groups[0]['lr']})
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))

        output_log.append('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        output_log.append('\n')
        print('-' * 89)
        #write_to_graph('Val/loss', val_loss, writer, epoch)
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
        scheduler.step(val_loss)

    with open(output_path+"_"+"log.txt", "w") as outfile:
        outfile.write("\n".join(output_log))
else:

    test_dataset = SkeletonsDataset(args.test_data,args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    # loading the model again to evaluate for the test set.
    checkpoint = torch.load(args.checkpoint)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    test_loss, test_acc = evaluate(model, test_loader, draw_img=True, visualize=False, out_path='./visual')
    print('=' * 89)
    print('| End of testing | test loss {:5.2f} | test ppl {:8.2f} | test acc {:8.2f}'.format(
        test_loss, math.exp(test_loss), test_acc))
    print('=' * 89)
