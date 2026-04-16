import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import argparse


class MNISTSequenceDataset(Dataset):
    def __init__(self, data_dir, train=True, time_step=7):
        self.dataset = datasets.MNIST(
            data_dir, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.time_step = time_step
        self.input_dim = 784 // time_step

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x.view(self.time_step, self.input_dim), y


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers=1,
                           bidirectional=False, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    parser = argparse.ArgumentParser(description='Train LSTM MNIST Classifier')
    parser.add_argument('--hidden-size', default=64, type=int, metavar='HS')
    parser.add_argument('--time-step', default=4, type=int, metavar='TS',
                        help='must divide 784 evenly (e.g. 7, 14, 28)')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--data-dir', default='./data/mnist', type=str)
    parser.add_argument('--save-dir', default='../models/mnist_lstm/', type=str, metavar='SD')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    assert 784 % args.time_step == 0, \
        f'time_step {args.time_step} must divide 784 evenly'

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print('use_cuda:', device.type == 'cuda')

    train_dataset = MNISTSequenceDataset(args.data_dir, train=True,  time_step=args.time_step)
    test_dataset  = MNISTSequenceDataset(args.data_dir, train=False, time_step=args.time_step)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = LSTMClassifier(train_dataset.input_dim, args.hidden_size, 10, args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_dir = os.path.join(args.save_dir, f'lstm_{args.time_step}_{args.hidden_size}')

    for i in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        loss, acc = evaluate(model, test_loader, device)
        scheduler.step()
        print(f'Epoch {i+1:3d}  '
              f'train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%  '
              f'test_loss={loss:.4f} test_acc={acc*100:.2f}%')

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'lstm'))
    print(f'Saved model to {save_dir}/lstm')
