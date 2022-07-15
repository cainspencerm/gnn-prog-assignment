from models import cnn
from data import mnist_mini

import torch
from torch import nn
from torch.utils.data import DataLoader
from ignite import engine, metrics

torch.manual_seed(42)


def test(batch_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = cnn.Classifier()
    model.load_state_dict(torch.load('checkpoints/cnn_epochs_6_batch_size_32_learning_rate_5e-03.pt'))
    model = model.to(device)

    test_set = mnist_mini.MNIST(data_dir='data', split='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Initialize metrics.
    def eval_step(engine_, batch):
        return batch

    default_evaluator = engine.Engine(eval_step)

    accuracy_score = metrics.Accuracy()
    accuracy_score.attach(default_evaluator, 'accuracy')

    criterion = nn.CrossEntropyLoss()

    # Evaluate the model.
    model.eval()

    with torch.no_grad():
        val_accuracy = 0.
        val_loss = 0.
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
        
            state = default_evaluator.run([[outputs, labels]])
            val_accuracy += state.metrics['accuracy']

    return val_loss / len(test_loader), val_accuracy / len(test_loader)


def main():
    val_loss, val_accuracy = test(batch_size=cnn.defaults['batch_size'])

    print(f'Validation loss: {val_loss}')
    print(f'Validation accuracy: {val_accuracy}')


if __name__ == '__main__':
    main()