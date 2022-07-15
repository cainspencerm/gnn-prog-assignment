import dgl.nn as dglnn
from models import spatial, spectral
from data import mnist_mini
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse



def train(epochs, batch_size, learning_rate, return_model=False):
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
        print("GAT model Spatial fiters running...")
        model = spatial.Classifier().to(device)
    elif args.model == 'spectral':
        print("GCN model Spectral fiters running...")
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

    #Defining evaluation metrices
    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)


    #Initializinf optimization function
    opt = torch.optim.Adam(model.parameters())

    #Begin training
    for epoch in range(epochs):
        model.train()
        # forward propagation by using all nodes
        logits = model(graph, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        val_loss = loss.item()
        val_accuracy = acc.item()

    if return_model:
        return val_loss, val_accuracy, model
    else:
        return val_loss, val_accuracy


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--param-search', action='store_true', help='Whether to search for best parameters')
    args = parser.parse_args()

    if args.param_search:
        state_results = []

        epoch_options = [4, 6, 8, 10]
        batch_options = [32, 64, 128, 256]
        learning_rate_options = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    for epochs in epoch_options:
        for batch_size in batch_options:
            for learning_rate in learning_rate_options:
                val_losses, val_accuracies = [], []
                for _ in range(3):
                    val_loss, val_accuracy = train(epochs, batch_size, learning_rate)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)

                state_results.append({'epochs': epochs, 'batch_size': batch_size, 'lr': learning_rate,
                                      'loss': sum(val_losses) / 3, 'acc': sum(val_accuracies) / 3})
                print(state_results[-1])

    with open('param_search/gnn_state_results.txt', 'w') as f:
        f.write('epochs, batch_size, lr, loss, acc\n')
        for result in state_results:
            f.write(f'{result["epochs"]}, {result["batch_size"]}, {result["lr"]}, {result["loss"]}, {result["acc"]}\n')




if __name__ == '__main__':
    train()
