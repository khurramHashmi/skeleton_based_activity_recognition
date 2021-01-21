from torch.utils.data import Dataset
from trainPC import *
import argparse

def train(args):
    root = './'

    ## training procedure
    teacher_force = False
    fix_weight = True
    fix_state = False

    if fix_weight:
        network = 'FW'

    if fix_state:
        network = 'FS'

    if not fix_state and not fix_weight:
        network = 'O'

    # hyperparameter
    feature_length = 75
    hidden_size =1024
    batch_size = args.batch_size
    en_num_layers = 3
    de_num_layers = 1
    print_every = 1
    learning_rate = args.lr
    epoch = args.max_epochs

    dataset_train = NTUDataset(args.train_datapath,args.train_labelpath, use_mmap=False, transformed_data=args.transformed_data)
    dataset_eval = NTUDataset(args.val_datapath,args.val_labelpath, use_mmap=False, transformed_data=args.transformed_data)

    shuffle_dataset = True
    dataset_size_train = len(dataset_train)
    dataset_size_eval = len(dataset_eval)

    indices_train = list(range(dataset_size_train))
    indices_eval = list(range(dataset_size_eval))
    random_seed = 11111
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_eval)

    print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_eval)))
    # seperate train and validation
    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_eval)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler,collate_fn=pad_collate)
    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, sampler=valid_sampler,collate_fn=pad_collate)

    # # Training
    # load model
    model = seq2seq(feature_length, hidden_size, feature_length, batch_size,
                    en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force)
    # initilize weight
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    #                     nn.init.uniform_(param, a=-0.05, b=0.05)

    #check whether decoder gru weights are fixed
    if fix_weight:
        print(model.decoder.gru.requires_grad)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    criterion_seq = nn.L1Loss(reduction='none')
# Testing piece of code
    #
    # zero_count = 0
    # for ind, (eval_data, seq_len, label) in enumerate(eval_loader):
    #     # print(eval_data.shape)
    #     # print(seq_len)
    #     # print(label)
    #     if 0 in seq_len:
    #         zero_count += 1
    # print("ZERO COUNT :",zero_count)
    # # if ind >= 0:
    # import sys
    #
    # sys.exit(0)
    # Testing piece of code ends here

    if not os.path.exists(os.path.join(root,"output")):
        os.mkdir(os.path.join(root,"output"))

    file_output = open(root+'output/%sen%d_hid%d.txt'% (network, en_num_layers, hidden_size), 'w')

    training(epoch, train_loader, eval_loader, print_every,
                 model, optimizer, criterion_seq,  file_output,
                 root, network, en_num_layers, hidden_size,load_saved=args.is_resume, num_class=args.num_class, inference=args.inference)

    file_output.close()

if __name__ == '__main__':
    # NOTE:Only supports joint data

    parser = argparse.ArgumentParser(description='Skeleton based Activity Recognition')

    parser.add_argument('-tp', '--train_datapath', help='location of train dataset numpy file', required=True)
    parser.add_argument('-tl', '--train_labelpath', help='location of train label pickle file', required=True)
    parser.add_argument('-vp', '--val_datapath', help='location of Validation dataset numpy file', required=True)
    parser.add_argument('-vl', '--val_labelpath', help='location of Validation label pickle file', required=True)
    parser.add_argument('-td', '--transformed_data', help='True if transformed data is used instead of standard', default=False)
    parser.add_argument('-e', '--max_epochs', help='number of epochs for training', default=100)
    parser.add_argument('-l', '--lr', type=float, help='learning rate value', default=0.001)
    parser.add_argument('-b', '--batch_size', help='learning rate value', default=32)
    parser.add_argument('-ir', '--is_resume', help='True or false for reloading checkpoint', default=False)
    parser.add_argument('-inf', '--inference', help='True or false for inference ', default=False)
    parser.add_argument('-nc', '--num_class', help='number of classes for the dataset', default=60)


    args = parser.parse_args()

    train(args)
