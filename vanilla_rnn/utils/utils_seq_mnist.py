import torch
import numpy as np
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset

class MNISTSequenceDataset(Dataset):
    def __init__(self, data_dir, train=False, max_len=None):
        """
        Args:
            data_dir: 包含 trainimg-*/testimg-* 檔案的資料夾路徑
            train: True for training set, False for test set
            max_len: 最大序列長度（padding/truncate）
        """
        prefix = 'train' if train else 'test'
        npz_file = os.path.join(data_dir, f'{prefix}_preprocessed.npz')
        
        # 優先使用預處理檔案
        if os.path.exists(npz_file):
            print(f"Loading from preprocessed file: {npz_file}")
            data = np.load(npz_file, allow_pickle=True)
            self.sequences = list(data['sequences'])
            self.labels = list(data['labels'])
        else:
            # 回退到原始讀取
            print(f"Preprocessed file not found, loading from raw files...")
            self.sequences = []
            self.labels = []
            
            prefix_raw = 'trainimg' if train else 'testimg'
            pattern = os.path.join(data_dir, f'{prefix_raw}-*-inputdata.txt')
            input_files = sorted(glob.glob(pattern))
            
            print(f"Found {len(input_files)} {prefix_raw} samples")
            
            for input_file in tqdm(input_files, desc=f"Loading {prefix_raw}"):
                target_file = input_file.replace('-inputdata.txt', '-targetdata.txt')
                
                if not os.path.exists(target_file):
                    print(f"Warning: {target_file} not found, skipping")
                    continue
                
                raw_input = np.loadtxt(input_file)
                sequence_data = raw_input[2:].reshape(-1, 4)
                valid_mask = (sequence_data[:, 0] != -1)
                seq = sequence_data[valid_mask, :3] # 暫時修成:3, 原為:2
                
                raw_target = np.loadtxt(target_file)
                label = np.argmax(raw_target[0, :10])
                
                self.sequences.append(seq)
                self.labels.append(label)
        
        # 計算序列長度統計
        self.seq_lens = [len(s) for s in self.sequences]
        if max_len is None:
            self.max_len = max(self.seq_lens)
        else:
            self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Padding或截斷
        if len(seq) < self.max_len:
            padded = np.zeros((self.max_len, 3)) # 暫時修成:3, 原為:2
            padded[:len(seq)] = seq
        else:
            padded = seq[:self.max_len]
        
        return torch.FloatTensor(padded), torch.LongTensor([label])[0]
    
    def get_stats(self):
        lens = np.array(self.seq_lens)
        return {
            'mean': lens.mean(),
            'std': lens.std(),
            'max': lens.max(),
            'min': lens.min(),
            'median': np.median(lens)
        }