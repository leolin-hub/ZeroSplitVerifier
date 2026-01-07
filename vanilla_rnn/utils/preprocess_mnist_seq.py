import numpy as np
import glob
import os

def preprocess(data_dir, output_file, prefix):
    pattern = os.path.join(data_dir, f'{prefix}-*-inputdata.txt')
    input_files = sorted(glob.glob(pattern))
    
    all_seqs = []
    all_labels = []
    
    for i, input_file in enumerate(input_files):
        if (i + 1) % 1000 == 0:
            print(f"Processing {i+1}/{len(input_files)}")
        
        target_file = input_file.replace('-inputdata.txt', '-targetdata.txt')
        
        raw_input = np.loadtxt(input_file)
        raw_target = np.loadtxt(target_file)
        
        seq_data = raw_input[2:].reshape(-1, 4)
        valid_mask = (seq_data[:, 0] != -1)
        seq = seq_data[valid_mask, :3] # 暫時修成:3 原為:2
        label = np.argmax(raw_target[0, :10])  # 修正版
        
        all_seqs.append(seq)
        all_labels.append(label)
    
    np.savez_compressed(output_file, sequences=np.array(all_seqs, dtype=object), labels=np.array(all_labels, dtype=np.int64))
    print(f"Saved {len(all_seqs)} samples to {output_file}")

if __name__ == '__main__':
    data_dir = './data/mnist_seq/sequences'
    preprocess(data_dir, f'{data_dir}/test_preprocessed.npz', 'testimg')
    preprocess(data_dir, f'{data_dir}/train_preprocessed.npz', 'trainimg')