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
        if x is None or y is None:
            data_loader = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=train, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ])),
                batch_size=N, shuffle=shuffle)
            iterator = iter(data_loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        x = x.view(N, seq_len, -1)
        # num = N
        rand_label = torch.randint(num_labels-1, [N], dtype=torch.long, device=device) + 1
        #range from 1 to num_labels-1
        target_label = torch.fmod(y+rand_label, num_labels)
        if not rnn is None:
            out = rnn(x)
            pred = out.argmax(dim=1)
            idx = (pred == y)
            # num = idx.sum()
            x = x[idx]
            y = y[idx]
            target_label = target_label[idx]
            print('remained fraction: %.4f' % idx.float().mean())

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
