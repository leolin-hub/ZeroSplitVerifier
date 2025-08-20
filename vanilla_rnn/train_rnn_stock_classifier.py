import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import bound_vanilla_rnn as v_rnn
from utils.sample_stock_data import prepare_stock_tensors_split
from collections import Counter
import argparse
import os

def train_stock_rnn(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t Train Loss: {loss.item():.6f}')

def test_stock_rnn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    print(f"Prediction distribution: {Counter(all_preds)}")
    print(f"Target distribution: {Counter(all_targets)}")

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train Stock RNN Classifier')
    parser.add_argument('--csv-path', default='C:/Users/zxczx/POPQORN/vanilla_rnn/utils/A1_bin.csv', type=str)
    parser.add_argument('--window-size', default=10, type=int)
    parser.add_argument('--hidden-size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--train-ratio', default=0.8, type=float)
    parser.add_argument('--save-dir', default='../../models/stock_classifier/', type=str)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # 載入數據
    X_train, y_train, target_train, X_test, y_test, target_test = prepare_stock_tensors_split(
        args.csv_path, args.window_size, args.train_ratio, device
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 檢查標籤分布
    train_labels = y_train.cpu().numpy()
    test_labels = y_test.cpu().numpy()

    print("Training label distribution:", Counter(train_labels))
    print("Test label distribution:", Counter(test_labels))
    
    # 創建DataLoader
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    # 創建模型
    input_size = 1
    hidden_size = args.hidden_size
    output_size = 3
    time_step = args.window_size
    
    model = v_rnn.RNN(input_size, hidden_size, output_size, time_step, args.activation)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 訓練
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train_stock_rnn(model, device, train_loader, optimizer, epoch)
        accuracy = test_stock_rnn(model, device, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            # 建立正確的folder
            model_name = f'stock_rnn_{time_step}_{hidden_size}_{args.activation}'
            save_dir = os.path.join(args.save_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'rnn'))
            model.to(device)
            print(f"Model saved to {save_dir}/rnn")
    
    print(f'Best accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()