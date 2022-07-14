from models import cnn
import mnist_mini

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ignite import engine, metrics

import argparse


def train(epochs, batch_size, learning_rate, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

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
    model.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_accuracy = 0.
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            state = default_evaluator.run([[outputs, labels]])
            train_accuracy += state.metrics['accuracy']

        # Evaluate the model.
        model.eval()

        val_loss = 0.
        val_accuracy = 0.
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            
                state = default_evaluator.run([[outputs, labels]])
                val_accuracy += state.metrics['accuracy']

        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        return val_loss, val_accuracy


def create_model(model_type):
    if model_type == 'cnn':
        from models import cnn
        model = cnn.Classifier()
    elif model_type == 'spectral':
        from models import spectral
        model = spectral.Classifier()
    elif model_type == 'spatial':
        from models import spatial
        model = spatial.Classifier()
    else:
        raise ValueError(f'Model {model_type} not recognized')

    return model


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--model', type=str, default='cnn', help='The type of model: cnn, spectral, or spatial')
    parser.add_argument('--param-search', action='store_true', help='Whether to search for best parameters')
    args = parser.parse_args()

    if args.param_search:
        state_results = []

        epoch_options = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        batch_options = [32, 64, 128, 256]
        learning_rate_options = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

        for epochs in epoch_options:
            for batch_size in batch_options:
                for learning_rate in learning_rate_options:
                    val_losses, val_accuracies = [], []
                    for _ in range(3):
                        model = create_model(args.model)

                        val_loss, val_accuracy = train(epochs, batch_size, learning_rate, model)
                        val_losses.append(val_loss)
                        val_accuracies.append(val_accuracy)

                    state_results.append({'epochs': epochs, 'batch_size': batch_size, 'lr': learning_rate,
                        'loss': sum(val_losses) / 3, 'acc': sum(val_accuracies) / 3})
                    print(state_results[-1])
        
        with open(f'param_search/{args.model}_state_results.txt', 'w') as f:
            f.write('epochs, batch_size, lr, loss, acc\n')
            for result in state_results:
                f.write(f'{result["epochs"]}, {result["batch_size"]}, {result["lr"]}, {result["loss"]}, {result["acc"]}\n')
    else:
        model = create_model(args.model)
        
        val_loss, val_accuracy = train(
            epochs=model.defaults['epochs'],
            batch_size=model.defaults['batch_size'],
            learning_rate=model.defaults['learning_rate'],
            model=model
        )

        print(f'Validation loss: {val_loss}')
        print(f'Validation accuracy: {val_accuracy}')

        torch.save(model.state_dict(), f'checkpoints/{args.model}_{model.get_defaults()}.pt')



if __name__ == '__main__':
    main()