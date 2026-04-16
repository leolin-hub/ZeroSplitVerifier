#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a ReLU-LSTM classifier on MNIST (sequential).

Architecture changes vs. standard LSTM:
  - g gate : relu(yg)   instead of tanh(yg)
  - output : o_gate * c  instead of o_gate * tanh(c)

The rnn weight container is still nn.LSTM so My_relu_lstm.__init__(model.rnn)
can read weights without modification.

Saves to: ../models/mnist_relu_lstm/relu_lstm_{time_step}_{hidden_size}/relu_lstm
"""

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


# ---------------------------------------------------------------------------
# Dataset (identical to train_mnist_lstm.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ReLU-LSTM classifier
# ---------------------------------------------------------------------------

class ReLULSTMClassifier(nn.Module):
    """
    LSTM classifier with two activation changes:
      g gate : relu(yg)   (was tanh)
      output : o_gate * c  (was o_gate * tanh(c))

    self.rnn is a plain nn.LSTM used only as a weight container.
    Its weight attributes (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
    input_size, hidden_size) are read directly by My_relu_lstm.__init__.
    """

    def __init__(self, input_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers=1,
                           bidirectional=False, dropout=0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        hid = self.rnn.hidden_size

        h = torch.zeros(batch, hid, device=x.device)
        c = torch.zeros(batch, hid, device=x.device)

        W_ih = self.rnn.weight_ih_l0   # (4h, input)
        W_hh = self.rnn.weight_hh_l0   # (4h, h)
        b_ih = self.rnn.bias_ih_l0     # (4h,)
        b_hh = self.rnn.bias_hh_l0     # (4h,)

        for t in range(seq_len):
            gates = x[:, t] @ W_ih.t() + b_ih + h @ W_hh.t() + b_hh
            # nn.LSTM gate order: i, f, g, o
            yi = gates[:, 0*hid:1*hid]
            yf = gates[:, 1*hid:2*hid]
            yg = gates[:, 2*hid:3*hid]
            yo = gates[:, 3*hid:4*hid]

            i_gate = torch.sigmoid(yi)
            f_gate = torch.sigmoid(yf)
            g_gate = torch.relu(yg)        # ← relu instead of tanh
            o_gate = torch.sigmoid(yo)

            c = f_gate * c + i_gate * g_gate
            h = o_gate * c                 # ← identity on c, no tanh

        return self.fc(self.dropout(h))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

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
        epoch_loss    += loss.item() * target.size(0)
        epoch_correct += pred.eq(target).sum().item()
        epoch_total   += target.size(0)
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
            epoch_loss    += loss.item() * target.size(0)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total   += target.size(0)
    return epoch_loss / epoch_total, epoch_correct / epoch_total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ReLU-LSTM MNIST Classifier')
    parser.add_argument('--hidden-size', default=64,  type=int,   metavar='HS')
    parser.add_argument('--time-step',   default=4,   type=int,   metavar='TS',
                        help='must divide 784 evenly (e.g. 4, 7, 14, 28)')
    parser.add_argument('--dropout',     default=0.5, type=float)
    parser.add_argument('--data-dir',    default='./data/mnist', type=str)
    parser.add_argument('--save-dir',    default='../models/mnist_relu_lstm/', type=str)
    parser.add_argument('--epochs',      default=50,  type=int)
    parser.add_argument('--batch-size',  default=128, type=int)
    parser.add_argument('--lr',          default=1e-3, type=float)
    parser.add_argument('--cuda',        action='store_true')
    args = parser.parse_args()

    assert 784 % args.time_step == 0, \
        f'time_step {args.time_step} must divide 784 evenly'

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_dataset = MNISTSequenceDataset(args.data_dir, train=True,  time_step=args.time_step)
    test_dataset  = MNISTSequenceDataset(args.data_dir, train=False, time_step=args.time_step)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = ReLULSTMClassifier(
        train_dataset.input_dim, args.hidden_size, 10, args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_dir = os.path.join(
        args.save_dir,
        f'relu_lstm_{args.time_step}_{args.hidden_size}'
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, device)
        scheduler.step()
        print(f'Epoch {epoch+1:3d}  '
              f'train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%  '
              f'test_loss={test_loss:.4f}  test_acc={test_acc*100:.2f}%')

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'relu_lstm'))
    print(f'Saved model to {save_dir}/relu_lstm')
