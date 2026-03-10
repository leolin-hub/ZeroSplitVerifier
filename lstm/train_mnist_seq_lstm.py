import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from vanilla_rnn.utils.utils_seq_mnist import MNISTSequenceDataset


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers=1,
                           bidirectional=False, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, time_step, input_size]
        _, (hidden, _) = self.rnn(x)
        hidden = self.dropout(hidden[0])
        return self.fc(hidden)


def train(model, loader, optimizer, device):
    epoch_loss = epoch_correct = epoch_total = 0
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        loss = F.cross_entropy(output, target, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * target.size(0)
        epoch_correct += pred.eq(target).sum().item()
        epoch_total += target.size(0)
    return epoch_loss / epoch_total, epoch_correct / epoch_total


def evaluate(model, loader, device):
    epoch_loss = epoch_correct = epoch_total = 0
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            loss = F.cross_entropy(output, target, reduction='mean')
            epoch_loss += loss.item() * target.size(0)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total += target.size(0)
    return epoch_loss / epoch_total, epoch_correct / epoch_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM MNIST-Sequence Classifier')
    parser.add_argument('--hidden-size', default=32, type=int, metavar='HS')
    parser.add_argument('--time-step', default=55, type=int, metavar='TS',
                        help='max sequence length (default: 55)')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--data-dir', default='./data/mnist_seq/sequences/', type=str)
    parser.add_argument('--save-dir', default='../models/mnist_seq_lstm/', type=str, metavar='SD')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('use_cuda:', use_cuda)

    train_dataset = MNISTSequenceDataset(args.data_dir, train=True, max_len=args.time_step)
    test_dataset = MNISTSequenceDataset(args.data_dir, train=False, max_len=args.time_step)

    stats = train_dataset.get_stats()
    print(f"Sequence stats: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
          f"max={stats['max']}, median={stats['median']:.1f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    INPUT_SIZE = 3  # dx, dy, pen state
    model = LSTMClassifier(INPUT_SIZE, args.hidden_size, 10, args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    save_dir = os.path.join(args.save_dir, f'lstm_hidden_size_{args.hidden_size}_ts_{args.time_step}')

    for i in range(args.epochs):
        print(f'Epoch {i+1} Train')
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        loss, acc = evaluate(model, test_loader, device)
        scheduler.step()
        print('-----------------------------------------------')
        print(f'Epoch {i+1} Testset loss: {loss:.3f} accuracy: {acc*100:.2f}')
        print('-----------------------------------------------')

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'lstm'))
    print(f'Have saved the trained model to {save_dir}/lstm')
