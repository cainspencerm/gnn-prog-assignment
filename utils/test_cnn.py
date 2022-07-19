from bitarray import test
from models import cnn
from data import mnist_mini

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics as metrics
import numpy as np
from torch_geometric.nn import models
import networkx as nx

import argparse


def test(model, batch_size=cnn.defaults['batch_size']):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare dataset.
    test_set = mnist_mini.MNIST(data_dir='data', split='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Evaluate the model.
    model.eval()

    with torch.no_grad():
        test_acc = 0.
        test_loss = 0.
        conf_mat = np.zeros((10, 10), dtype=int)
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Compute accuracy.
            _, indices = torch.max(outputs, dim=1)
            correct = torch.sum(indices == labels)
            test_acc += correct.item() * 1.0 / len(data)

            # Compute confusion matrix.
            conf_mat += metrics.confusion_matrix(labels.cpu().numpy(), indices.cpu().numpy(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    return test_loss, test_acc, conf_mat


def correct_and_smooth(model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = mnist_mini.MNIST_Graph()
    data = dataset[0]
    node_features = data.ndata['feat'].to(device)
    node_labels = data.ndata['label'].to(device)
    test_mask = data.ndata['test_mask'].to(device)

    model = model.to(device)
    model.eval()

    post = models.CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0,
            num_smoothing_layers=50, smoothing_alpha=0.8,
            autoscale=False, scale=20.)

    with torch.no_grad():
        node_features = node_features.view(-1, 1, 16, 16)

        outputs = model(node_features)
        outputs = outputs.softmax(dim=1)

        y_soft = post.correct(outputs, node_labels[test_mask].long(), test_mask, dataset.edge_index().to(device))
        y_soft = post.smooth(y_soft, node_labels[test_mask].long(), test_mask, dataset.edge_index().to(device))

        test_acc = torch.sum(y_soft.argmax(dim=1)[test_mask] == node_labels[test_mask]).item() / torch.sum(test_mask).item()

        conf_mat = metrics.confusion_matrix(y_soft.argmax(dim=1)[test_mask].cpu().numpy(), node_labels[test_mask].cpu().numpy(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    return test_acc, conf_mat


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--c-s', action='store_true', help='Whether to use correct and smooth')
    parser.add_argument('--weights', default=f'checkpoints/cnn_{cnn.get_defaults()}.pt', type=str, help='Path to weights')
    args = parser.parse_args()

    model = cnn.Classifier()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    if args.c_s:
        test_acc, conf_mat = correct_and_smooth(model)

        print(f'C&S Test Accuracy: {test_acc:.5f}')
        print(f'Confusion Matrix: \n{conf_mat}')
    
    else:
        test_loss, test_acc, conf_mat = test(model)

        print(f'Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.5f}')
        print(f'Confusion Matrix: \n{conf_mat}')


if __name__ == '__main__':
    main()