from models import spatial, spectral
from data import mnist_mini

import torch
import torch.nn as nn
import networkx as nx
import argparse
from torch.utils import tensorboard as tb
import dgl
from sklearn import metrics


def test(model, model_type, data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph, dataset, adj_mat, node_features, node_labels, train_mask, test_mask = data
    graph = graph.to(device)
    adj_mat = adj_mat.to(device)
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)

    model = model.to(device)

    if model_type is spectral:
        graph_rep = adj_mat

    elif model_type is spatial:
        graph_rep = graph
    
    criterion = nn.CrossEntropyLoss()

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

    return test_loss, test_acc, conf_mat


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--model', required=True, type=str, help='Model type to use')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights')
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
        model = spatial.Classifier()
        model.load_state_dict(torch.load(args.weights))

        model_type = spatial

        # Add self loops to graph.
        graph = dgl.add_self_loop(graph)
    
    elif args.model == 'spectral':
        model = spectral.Classifier()
        model.load_state_dict(torch.load(args.weights))

        model_type = spectral

    else:
        raise ValueError('Unknown model type')

    data = graph, dataset, adj_mat, node_features, node_labels, train_mask, test_mask

    test_loss, test_acc, conf_mat = test(model, model_type=model_type, data=data)

    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{conf_mat}')


if __name__ == '__main__':
    main()