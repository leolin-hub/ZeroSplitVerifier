import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from bound_vanilla_rnn import RNN
from utils.utils_cifar10 import CIFAR10SequenceDataset
from torch.utils.data import DataLoader

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.sampler)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    print(f'Test: Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')
    return acc

def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_dataset = CIFAR10SequenceDataset(
        args.data_dir, train=True, time_step=args.time_step, use_rgb=args.use_rgb
    )
    test_dataset = CIFAR10SequenceDataset(
        args.data_dir, train=False, time_step=args.time_step, use_rgb=args.use_rgb
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    input_size = train_dataset.input_dim
    model = RNN(input_size, args.hidden_size, 10, args.time_step, args.activation)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        
        if acc > best_acc:
            best_acc = acc
            model_name = f'rnn_{args.time_step}_{args.hidden_size}_{args.activation}'
            save_dir = os.path.join(args.save_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'rnn'))
            model.to(device)
            print(f"Saved best model (acc={best_acc:.2f}%)")
    
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', default=64, type=int)
    parser.add_argument('--time-step', default=32, type=int)
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--save-dir', default='../models/cifar10_classifier/', type=str)
    parser.add_argument('--use-rgb', default=True, type=bool)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--test-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--log-interval', default=100, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    main(args)