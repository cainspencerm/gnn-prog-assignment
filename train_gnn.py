from models import spatial, spectral
from data import mnist_mini

import torch
import torch.nn as nn
import networkx as nx
import argparse
from torch.utils import tensorboard as tb
import dgl
from sklearn import metrics


def train(learning_rate, model_type, data, epoch_options: list, return_model=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph, dataset, adj_mat, node_features, node_labels, train_mask, test_mask = data
    graph = graph.to(device)
    adj_mat = adj_mat.to(device)
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)

    # Create the model.
    model = model_type.Classifier().to(device)

    if model_type is spectral:
        model_name = 'spectral'
        graph_rep = adj_mat

    elif model_type is spatial:
        model_name = 'spatial'
        graph_rep = graph

    else:
        raise ValueError('Unknown model type')
    
    writer = tb.SummaryWriter(f'runs/gnn_{model_name}_{model_type.get_defaults()}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Begin training.
    for epoch in range(max(epoch_options)):

        model.train()

        # Forward pass.
        outputs = model(graph_rep, node_features)
        loss = criterion(outputs[train_mask], node_labels[train_mask])

        # Backward and optimize.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, indices = torch.max(outputs, dim=1)
        correct = torch.sum(indices[train_mask] == node_labels[train_mask])
        train_acc = correct.item() * 1.0 / dataset.get_num_nodes(split='train')

        # Validate models.
        model.eval()

        with torch.no_grad():
            outputs = model(graph_rep, node_features)
            test_loss = criterion(outputs[test_mask], node_labels[test_mask])

            # Compute accuracy.
            _, indices = torch.max(outputs, dim=1)
            correct = torch.sum(indices[test_mask] == node_labels[test_mask])
            test_acc = correct.item() * 1.0 / dataset.get_num_nodes(split='test')

            # Create the confusion matrix.
            conf_mat = metrics.confusion_matrix(node_labels[test_mask].cpu().numpy(), indices[test_mask].cpu().numpy())

        if epoch + 1 in epoch_options:
            print(f'Epoch: {epoch + 1:02} | Learning Rate: {learning_rate:.0e} | Train Loss: {loss.item():.4f} | Val. Loss: {test_loss.item():.4f} | Val. Acc: {test_acc:.2f}')
            print(f'Confusion Matrix: \n{conf_mat}')

        writer.add_scalar('Train/Loss', loss.item(), epoch + 1)
        writer.add_scalar('Train/Acc', train_acc, epoch + 1)
        writer.add_scalar('Test/Loss', test_loss.item(), epoch + 1)
        writer.add_scalar('Test/Acc', test_acc, epoch + 1)
    
        writer.flush()

    writer.close()

    if return_model:
        return test_loss, test_acc, model
    else:
        return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--model', type=str, help='Model type to use')
    parser.add_argument('--param-search', action='store_true', help='Whether to search for best parameters')
    args = parser.parse_args()

    # Prepare the data.
    dataset = mnist_mini.MNIST_Graph()
    g = nx.Graph()
    g.add_edges_from(dataset._edges)
    adj_mat = torch.tensor(nx.adjacency_matrix(g).todense(), dtype=torch.float)

    graph = dataset[0]
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']

    if args.model == 'spatial':
        model_module = spatial

        # Add self loops to graph.
        graph = dgl.add_self_loop(graph)
    
    elif args.model == 'spectral':
        model_module = spectral

    else:
        raise ValueError('Unknown model type')

    data = graph, dataset, adj_mat, node_features, node_labels, train_mask, test_mask

    if args.param_search:
        epoch_options = [6, 8, 10, 12, 14, 16, 18, 20]
        learning_rate_options = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
        for learning_rate in learning_rate_options:
            val_loss, val_accuracy = train(learning_rate, model_module, data, epoch_options=epoch_options)

    else:
        val_loss, val_accuracy, model = train(
            learning_rate=model_module.defaults['learning_rate'],
            model_type=model_module,
            data=data,
            epoch_options=[model_module.defaults['epochs']],
            return_model=True
        )

        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.2f}')

        torch.save(model.state_dict(), f'checkpoints/gnn_{model_module.get_defaults()}.pt')


if __name__ == '__main__':
    main()