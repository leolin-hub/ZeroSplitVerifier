from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10SequenceDataset(Dataset):
    def __init__(self, data_dir='./data', train=True, time_step=None, use_rgb=True):
        """
        Args:
            data_dir: CIFAR-10 下載路徑
            train: True for train, False for test
            time_step: 序列長度（必須整除 3072 或 1024）
            use_rgb: True 保留 RGB (3072), False 轉灰階 (1024)
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if use_rgb 
            else transforms.Grayscale(),
        ])
        
        self.dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
        self.use_rgb = use_rgb
        self.total_dim = 3072 if use_rgb else 1024
        
        if time_step is None:
            self.time_step = self.total_dim
        else:
            assert self.total_dim % time_step == 0, \
                f"time_step={time_step} 必須整除 {self.total_dim}"
            self.time_step = time_step
        
        self.input_dim = self.total_dim // self.time_step
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # 展平: [C,H,W] -> [T, D]
        flat = img.flatten()  # [3072] or [1024]
        seq = flat.view(self.time_step, self.input_dim)
        
        return seq, label