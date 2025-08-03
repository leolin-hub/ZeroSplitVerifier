import torch
import numpy as np
from bound_vanilla_rnn import RNN
from zerosplit_verifier import ZeroSplitVerifier
from utils.sample_data import sample_mnist_data

def test_bounds_consistency(input_size=None, hidden_size=2, output_size=10, time_step=2, p=2, batch_size=5, eps=0.5, activation='relu', model_path='C:/Users/LiLe556/Leo file/models/mnist_classifier/rnn_2_2_relu/rnn'):
    device = torch.device('cpu')
    
    if input_size is None:
        input_size = int(28 * 28 / time_step)
    
    # 分別建立兩個驗證器，rnn是原本，verifier是改寫過後
    rnn = RNN(input_size, hidden_size, output_size, time_step, activation).to(device)
    zsv = ZeroSplitVerifier(input_size, hidden_size, output_size, time_step, activation, max_splits=3, debug=True)
    
    # 載入同樣權重
    rnn.load_state_dict(torch.load(model_path, map_location='cpu'))
    zsv.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    rnn.to(device)
    zsv.to(device)
    
    # Extract Weight
    rnn.extractWeight(clear_original_model=False)
    zsv.extractWeight(clear_original_model=False)
    
    X, y, target_label = sample_mnist_data(
        N=batch_size,
        seq_len=time_step,
        device=device,
        data_dir='../../data/mnist',
        train=False,
        shuffle=True,
    )
    
    # bound_vanilla_rnn.py
    yL, yU = rnn.getLastLayerBound(eps, p, X = X, clearIntermediateVariables=True)
    
    print('Have computed the last layer bound in bound_vanilla_rnn.py: ')
    print('yL: ', yL)
    print('yU: ', yU)
    
    # zerosplit_verifier.py
    zsv_yL_hidden, zsv_yU_hidden = zsv.computeHiddenBounds(eps, p=p, v=1, X=X, Eps_idx=torch.arange(1, time_step+1))
    
    print(f"Final hidden bounds in ZeroSplitVerifier:")
    print(f"Lower hidden bound: {zsv_yL_hidden}")
    print(f"Upper hidden bound: {zsv_yU_hidden}")
    
    zsv_yL, zsv_yU = zsv.computeLast2sideBound(eps, p=p, v=time_step+1, X=X, Eps_idx=torch.arange(1, time_step+1))
    print(f"Final output bounds in ZeroSplitVerifier:")
    print(f"Lower output bound: {zsv_yL}")
    print(f"Upper output bound: {zsv_yU}")
    
    # Compare bounds
    if torch.allclose(zsv_yL, yL) and torch.allclose(zsv_yU, yU):
        print("\nBounds are identical:")
        print("Lower bound:", yL)
        print("Upper bound:", yU)
    else:
        print("\nError: Bounds are different!")
        print("Difference in lower bounds:", torch.max(torch.abs(zsv_yL - yL)))
        print("Difference in upper bounds:", torch.max(torch.abs(zsv_yU - yU)))
        
if __name__ == '__main__':
    params = {
        'hidden_size': 2,
        'output_size': 10,
        'time_step': 2,
        'p': 2,
        'batch_size': 10,
        'eps': 0.5,
        'activation': 'relu',
        'model_path': 'C:/Users/LiLe556/Leo file/models/mnist_classifier/rnn_2_2_relu/rnn'
    }
    test_bounds_consistency(**params)