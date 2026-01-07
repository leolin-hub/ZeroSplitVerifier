import torch
import numpy as np
import random
import os
from .utils_seq_mnist import MNISTSequenceDataset
# 隨機模擬固定用
import random
import numpy as np

# Set random seed for reproducibility
seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def sample_seq_mnist_data(N, time_step, device, num_labels=10,
                          data_dir='./data/mnist_seq/sequences/', train=False, 
                          shuffle=True, rnn=None, x=None, y=None):
    """
    Sample N correctly predicted samples from MNIST sequence dataset
    
    Returns:
        x: [N, time_step, 2] input sequences
        y: [N] ground-truth labels
        target_label: [N] target labels for verification (different from y)
    """
    with torch.no_grad():
        if rnn is None:
            # 沒有模型時的原邏輯
            if x is None and y is None:
                dataset = MNISTSequenceDataset(data_dir, train=train, max_len=time_step)
                loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=shuffle)
                x, y = next(iter(loader))
            
            x, y = x.to(device), y.to(device)
            rand_label = torch.randint(num_labels-1, [len(y)], dtype=torch.long, device=device) + 1
            target_label = torch.fmod(y + rand_label, num_labels)
            return x, y, target_label
        
        # 有模型時：循環採樣直到獲得N個正確預測的樣本
        dataset = MNISTSequenceDataset(data_dir, train=train, max_len=time_step)
        loader = torch.utils.data.DataLoader(dataset, batch_size=N*2, shuffle=shuffle)
        
        collected_x, collected_y, collected_target = [], [], []
        total_sampled = 0
        total_checked = 0
        total_correct = 0 # new
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            rand_label = torch.randint(num_labels-1, [batch_x.size(0)], 
                                     dtype=torch.long, device=device) + 1
            batch_target = torch.fmod(batch_y + rand_label, num_labels)
            
            out = rnn(batch_x)
            pred = out.argmax(dim=1)
            # idx = (pred == batch_y)
            correct = (pred == batch_y)  # new
            
            # 收集正確預測的樣本
            collected_x.append(batch_x) # 將[idx]移除
            collected_y.append(batch_y) # 將[idx]移除
            collected_target.append(batch_target) # 將[idx]移除
            # 原先只收正確的樣本
            # total_sampled += idx.sum().item()
            # total_checked += len(idx)
            # 不管正確與否都收
            total_sampled += len(batch_x)
            total_checked += len(batch_x)
            total_correct += correct.sum().item() # new
            
            if total_sampled >= N:
                break
        
        # 合併並截取前N個
        x = torch.cat(collected_x, dim=0)[:N]
        y = torch.cat(collected_y, dim=0)[:N]
        target_label = torch.cat(collected_target, dim=0)[:N]
        
        # accuracy = total_sampled / total_checked if total_checked > 0 else 0
        accuracy = total_correct / total_checked if total_checked > 0 else 0 # new
        print(f'Correctly predicted: {total_correct}/{total_checked} ({100*accuracy:.1f}%), Collected {len(x)} samples')
    
    return x, y, target_label

if __name__ == '__main__':
    # 測試
    import numpy as np
    test_file = './data/mnist_seq/sequences/testimg-0-targetdata.txt'
    raw = np.loadtxt(test_file)
    
    print(f"raw shape: {raw.shape}")
    print(f"First 15 values: {raw[:15]}")
    print(f"First 10 (one-hot): {raw[:10]}")
    print(f"argmax: {np.argmax(raw[:10])}")
    print(f"Expected: 7 (from your example)")
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    N = 5
    time_step = 100
    device = torch.device('cpu')

    data_dir = './data/mnist_seq/sequences/'
    print(f"Checking directory: {os.path.abspath(data_dir)}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    
    x, y, target_label = sample_seq_mnist_data(
        N, time_step, device, train=False, shuffle=True, rnn=None
    )
    
    print(f"x.shape: {x.shape}")
    print(f"y: {y}")
    print(f"target_label: {target_label}")