import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
# 隨機模擬固定用
import random
import numpy as np

# Set random seed for reproducibility
seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def sample_mnist_data(N, seq_len, device, num_labels=10,
                      data_dir='../../data/mnist', train=False, shuffle=True,
                      rnn=None, x=None, y=None):
    with torch.no_grad():
        if rnn is None:
            # 沒有模型時的原邏輯
            if x is None and y is None:
                data_loader = torch.utils.data.DataLoader(
                    datasets.MNIST(data_dir, train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                ])),
                    batch_size=N, shuffle=shuffle)
                iterater = iter(data_loader)
                x,y = next(iterater)
            x,y = x.to(device), y.to(device)
            x = x.view(N, seq_len, -1)
            rand_label = torch.randint(num_labels-1,[N],dtype=torch.long, device=device) + 1
            target_label = torch.fmod(y+rand_label, num_labels)
            return x,y,target_label
        
        # 有模型時：循環採樣直到獲得N個正確預測的樣本
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=train, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=N*2, shuffle=shuffle)  # 取更多樣本
        
        collected_x, collected_y, collected_target = [], [], []
        total_sampled = 0
        
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.view(batch_x.size(0), seq_len, -1)
            
            rand_label = torch.randint(num_labels-1, [batch_x.size(0)], 
                                     dtype=torch.long, device=device) + 1
            batch_target = torch.fmod(batch_y + rand_label, num_labels)
            
            out = rnn(batch_x)
            pred = out.argmax(dim=1)
            idx = (pred == batch_y)
            
            # 收集正確預測的樣本
            collected_x.append(batch_x[idx])
            collected_y.append(batch_y[idx])
            collected_target.append(batch_target[idx])
            total_sampled += idx.sum().item()
            
            if total_sampled >= N:
                break
        
        # 合併並截取前N個
        x = torch.cat(collected_x, dim=0)[:N]
        y = torch.cat(collected_y, dim=0)[:N]
        target_label = torch.cat(collected_target, dim=0)[:N]
        
        print(f'Accuracy: {total_sampled/((len(collected_x)*batch_x.size(0)) or 1):.4f}, Got {len(x)} samples')
        
    return x, y, target_label

if __name__ == '__main__':
    N = 5
    seq_len = 2
    device = torch.device('cpu')
    # device = torch.device('cuda:0)
    # 測試隨機用
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x, y, target_label = sample_mnist_data(N, seq_len, device, num_labels=10,
        data_dir='../data/mnist', train=False, shuffle=True, rnn=None, x=None, y=None)
    print("x = ", x)
    print("y = ", y)
    print("target_label = ", target_label)