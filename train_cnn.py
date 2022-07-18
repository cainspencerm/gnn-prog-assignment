from models import cnn
from data import mnist_mini

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ignite import engine, metrics
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics as sk_metrics
import numpy as np

import argparse


def train(batch_size, learning_rate, epoch_options, return_model=False):

    writer = SummaryWriter(f'runs/cnn_{batch_size}_{learning_rate}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn.Classifier().to(device)

    # Prepare datasets.
    train_set = mnist_mini.MNIST(data_dir='data', split='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = mnist_mini.MNIST(data_dir='data', split='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Initialize metrics.
    def eval_step(engine_, batch):
        return batch

    default_evaluator = engine.Engine(eval_step)

    accuracy_score = metrics.Accuracy()
    accuracy_score.attach(default_evaluator, 'accuracy')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Begin training.
    for epoch in range(max(epoch_options)):
        model.train()

        train_loss = 0.
        train_acc = 0.
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            state = default_evaluator.run([[outputs, labels]])
            train_acc += state.metrics['accuracy']

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

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
                conf_mat += sk_metrics.confusion_matrix(labels.cpu().numpy(), indices.cpu().numpy(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            if epoch + 1 in epoch_options:
                print(f'Epoch: {epoch + 1:02} | Batch Size: {batch_size} | Learning Rate: {learning_rate:.0e} | Train Loss: {train_loss:.4f} | Val. Loss: {test_loss:.4f} | Val. Acc: {test_acc:.2f}')

                print(f'Confusion Matrix: \n{conf_mat}')

            writer.add_scalar('Train/Loss', train_loss, epoch + 1)
            writer.add_scalar('Train/Acc', train_acc, epoch + 1)
            writer.add_scalar('Test/Loss', test_loss, epoch + 1)
            writer.add_scalar('Test/Acc', test_acc, epoch + 1)

        writer.flush()

    writer.close()

    if return_model:
        return test_loss, test_acc, model
    else:
        return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--param-search', action='store_true', help='Whether to search for best parameters')
    args = parser.parse_args()

    if args.param_search:
        
        epoch_options = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        batch_options = [32, 64, 128, 256]
        learning_rate_options = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

        for batch_size in batch_options:
            for learning_rate in learning_rate_options:
                val_loss, val_accuracy = train(batch_size, learning_rate, epoch_options=epoch_options)

    else:

        val_loss, val_accuracy, model = train(
            batch_size=cnn.defaults['batch_size'],
            learning_rate=cnn.defaults['learning_rate'],
            epoch_options=[cnn.defaults['epochs']],
            return_model=True
        )

        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.2f}')

        torch.save(model.state_dict(), f'checkpoints/cnn_{cnn.get_defaults()}.pt')


if __name__ == '__main__':
    main()