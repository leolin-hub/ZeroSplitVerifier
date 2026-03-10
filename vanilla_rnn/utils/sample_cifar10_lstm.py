import torch
import numpy as np
import random
from vanilla_rnn.utils.utils_cifar10 import CIFAR10SequenceDataset

seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def sample_cifar10_data(N, time_step, device, num_labels=10,
                        data_dir='./data', train=False, use_rgb=True,
                        shuffle=True, rnn=None):
    """
    Sample N samples from CIFAR-10 (no correctness filtering).

    Returns:
        x:            [N, time_step, input_dim]
        y:            [N] ground-truth labels
        target_label: [N] labels != y
    """
    with torch.no_grad():
        dataset = CIFAR10SequenceDataset(
            data_dir, train=train, time_step=time_step, use_rgb=use_rgb
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(N, 256), shuffle=shuffle
        )

        collected_x, collected_y, collected_target = [], [], []
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            rand_label = torch.randint(num_labels - 1, [batch_x.size(0)],
                                       dtype=torch.long, device=device) + 1
            batch_target = torch.fmod(batch_y + rand_label, num_labels)

            collected_x.append(batch_x)
            collected_y.append(batch_y)
            collected_target.append(batch_target)
            total += batch_x.size(0)

            if total >= N:
                break

        x = torch.cat(collected_x, dim=0)[:N]
        y = torch.cat(collected_y, dim=0)[:N]
        target_label = torch.cat(collected_target, dim=0)[:N]

    return x, y, target_label
