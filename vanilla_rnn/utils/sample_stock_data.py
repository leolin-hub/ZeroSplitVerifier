import pandas as pd
import torch
import numpy as np
from utils.utils_stock import mark_labels

def load_stock_data(file_path):
    
    df = pd.read_csv(file_path)
    df_labeled, _ = mark_labels(df)
    return df_labeled

def encode_labels(labels):

    label_map = {'持有': 0, '買進': 1, '賣出': 2}
    return [label_map.get(label, 0) for label in labels]

def create_sliding_windows(data, window_size=10, train_end_idx=None):
    
    close_prices = data['Close'].values
    labels = data['Label'].values

    if train_end_idx is not None:
        train_prices = close_prices[:train_end_idx]
        mean_price = np.mean(train_prices)
        std_price = np.std(train_prices)
    else:
        # 如果沒指定，用前30%數據作為統計基準
        split_point = max(window_size, int(len(close_prices) * 0.3))
        mean_price = np.mean(close_prices[:split_point])
        std_price = np.std(close_prices[:split_point])
    
    normalize_prices = (close_prices - mean_price) / std_price

    sequences = []
    sequence_labels = []

    for i in range(len(normalize_prices) - window_size + 1):
        seq = normalize_prices[i:i + window_size]
        label = labels[i+window_size - 1]

        sequences.append(seq)
        sequence_labels.append(label)

    return np.array(sequences), np.array(sequence_labels), mean_price, std_price

def prepare_stock_tensors_split(csv_path, window_size=10, train_ratio=0.8, device='cpu'):
    
    stock_data = load_stock_data(csv_path)
    
    # 先分割原始數據
    total_len = len(stock_data)
    train_end_idx = int(total_len * train_ratio)
    
    # 用訓練集統計量做標準化
    sequences, labels, mean_price, std_price = create_sliding_windows(
        stock_data, window_size, train_end_idx=train_end_idx
    )
    
    encoded_labels = encode_labels(labels)
    
    # 轉換為tensor
    X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # [N, seq_len, 1]
    y = torch.tensor(encoded_labels, dtype=torch.long)

    # 分割sequences
    seq_train_end = train_end_idx - window_size + 1
    X_train = X[:seq_train_end]
    y_train = y[:seq_train_end]
    X_test = X[seq_train_end:]
    y_test = y[seq_train_end:]
    
    # 生成target_label (用於對抗性測試)
    target_train = torch.randint(0, 3, (len(y_train),), dtype=torch.long)
    mask = target_train == y_train
    target_train[mask] = (target_train[mask] + 1) % 3
    
    target_test = torch.randint(0, 3, (len(y_test),), dtype=torch.long)
    mask = target_test == y_test
    target_test[mask] = (target_test[mask] + 1) % 3
    
    return (X_train.to(device), y_train.to(device), target_train.to(device),
            X_test.to(device), y_test.to(device), target_test.to(device))

if __name__ == "__main__":
    csv_path = 'C:/Users/zxczx/POPQORN/vanilla_rnn/utils/A1_bin.csv'
    window_size = 10
    train_ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train, target_train_label, X_test, y_test, target_test_label = prepare_stock_tensors_split(csv_path, window_size, train_ratio, device)
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}, target_label shape: {target_train_label.shape}")
    print(f"Label distribution: {torch.bincount(y_train)}")
    print(f"Label percentages: {torch.bincount(y_train).float() / (len(y_train) + len(y_test)) * 100}")