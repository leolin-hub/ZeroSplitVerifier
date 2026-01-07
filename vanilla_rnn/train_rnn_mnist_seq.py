#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
# from rnn_classifier import RNNClassifier as RNN
from bound_vanilla_rnn import RNN
from utils.utils_seq_mnist import MNISTSequenceDataset
from torch.utils.data import DataLoader

def train(log_interval, model, device, train_loader, optimizer, epoch, activation):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # if activation == 'tanh':
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 載入資料集並獲取統計
    print("Loading datasets...")
    train_dataset = MNISTSequenceDataset(
        args.data_dir,
        train=True,
        max_len=args.time_step
    )
    test_dataset = MNISTSequenceDataset(
        args.data_dir,
        train=False,
        max_len=args.time_step
    )
    
    stats = train_dataset.get_stats()
    print(f"Sequence length stats: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
          f"max={stats['max']}, median={stats['median']:.1f}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # 建立模型
    input_size = 3  # dx, dy
    model = RNN(input_size, args.hidden_size, 10, args.time_step, args.activation)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 訓練
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_loader, optimizer, epoch, args.activation)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            model_name = f'rnn_seq_{args.time_step}_{args.hidden_size}_{args.activation}'
            save_dir = os.path.join(args.save_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'rnn'))
            model.to(device)
            print(f"Saved best model (acc={best_acc:.2f}%)")
    
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', default=64, type=int)
    parser.add_argument('--time-step', default=70, type=int, 
                        help='max sequence length to use')
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--data-dir', default='./data/mnist_seq/sequences/', type=str)
    parser.add_argument('--save-dir', default='../models/mnist_seq_classifier/', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--test-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float) 
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--log-interval', default=100, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    main(args)