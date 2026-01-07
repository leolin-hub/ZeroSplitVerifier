import torch
import numpy as np
import random
from .utils_cifar10 import CIFAR10SequenceDataset

seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def sample_cifar10_data(N, time_step, device, num_labels=10,
                        data_dir='./data', train=False, use_rgb=True,
                        shuffle=True, rnn=None, x=None, y=None):
    """
    Sample N correctly predicted samples from CIFAR-10
    
    Returns:
        x: [N, time_step, input_dim] sequences
        y: [N] ground-truth labels
        target_label: [N] target labels (y != target_label)
    """
    with torch.no_grad():
        if rnn is None:
            if x is None and y is None:
                dataset = CIFAR10SequenceDataset(
                    data_dir, train=train, time_step=time_step, use_rgb=use_rgb
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=shuffle)
                x, y = next(iter(loader))
            
            x, y = x.to(device), y.to(device)
            rand_label = torch.randint(num_labels-1, [len(y)], dtype=torch.long, device=device) + 1
            target_label = torch.fmod(y + rand_label, num_labels)
            return x, y, target_label
        
        dataset = CIFAR10SequenceDataset(
            data_dir, train=train, time_step=time_step, use_rgb=use_rgb
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=N*2, shuffle=shuffle)
        
        collected_x, collected_y, collected_target = [], [], []
        total_sampled = 0
        total_checked = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            rand_label = torch.randint(num_labels-1, [batch_x.size(0)], 
                                     dtype=torch.long, device=device) + 1
            batch_target = torch.fmod(batch_y + rand_label, num_labels)
            
            out = rnn(batch_x)
            pred = out.argmax(dim=1)
            idx = (pred == batch_y)
            
            collected_x.append(batch_x[idx])
            collected_y.append(batch_y[idx])
            collected_target.append(batch_target[idx])
            total_sampled += idx.sum().item()
            total_checked += len(idx)
            
            if total_sampled >= N:
                break
        
        x = torch.cat(collected_x, dim=0)[:N]
        y = torch.cat(collected_y, dim=0)[:N]
        target_label = torch.cat(collected_target, dim=0)[:N]
        
        accuracy = total_sampled / total_checked if total_checked > 0 else 0
        print(f'Correctly predicted: {total_sampled}/{total_checked} ({100*accuracy:.1f}%), Collected {len(x)} samples')
    
    return x, y, target_label