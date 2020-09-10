import errno

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

def write_to_graph(labels, value,writer,epoch):
    writer.add_scalar(labels, value, epoch)

def data_distribution(dataset): #function generating histograms
    return 1

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise