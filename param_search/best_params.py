import argparse


def main():
    parser = argparse.ArgumentParser(description='Search for best parameters')
    parser.add_argument('--file', type=str, default='state_results.txt', help='File containing state results')
    args = parser.parse_args()

    max_acc = 0.
    max_acc_state = None
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            epoch, batch_size, lr, loss, acc = line.split(', ')
            acc = float(acc)
            if max_acc < acc:
                max_acc = acc
                max_acc_state = {'epochs': int(epoch), 'batch_size': int(batch_size), 'lr': float(lr), 'loss': float(loss), 'acc': float(acc)}

    print(max_acc_state)


if __name__ == '__main__':
    main()