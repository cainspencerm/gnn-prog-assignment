import dgl.nn as dglnn
from models import spatial, spectral
from data import mnist_mini
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Selecting gnn model

    parser = argparse.ArgumentParser(description='Retrieving model type')
    parser.add_argument('model',
                           metavar='model',
                           type=str,
                           help='model to train')

    args = parser.parse_args()
    #print(args.model)

    if args.model == 'spatial':
        print(args)
        model = spatial.Classifier().to(device)
    elif args.model == 'spectral':
        model = spatial.Classifier().to(device)
    else:
        print("Incorrect model name given")


    #Retrieving data
    dataset = mnist_mini.MNIST_Graph()
    graph = dataset[0]

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

    print(n_features)




if __name__ == '__main__':
    train()
