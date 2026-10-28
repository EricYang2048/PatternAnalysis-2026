import os
import numpy as np

from dataset import get_lobster_timeseries_dataset
from modules import timegan

def main():

    # paths for dataset
    orderbook_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
    message_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv'

    # message_path = '/home/groups/comp3710/LOBSTER/message_amzn.csv'
    # orderbook_path = '/home/groups/comp3710/LOBSTER/orderbook_amzn.csv'

    # parameters for timeGAN
    parameters = {
        'hidden_dim': 32,
        'num_layer': 3,
        'iterations': 10000, 
        'batch_size': 128,
        'module': 'gru',  
    }
    
    seq_len = 100 # num of timesteps
    train_rate = 0.8
    
    # load and preprocess LOBSTER dataset
    train_data, test_data, train_time, test_time = get_lobster_timeseries_dataset(
        message_path=message_path, 
        orderbook_path=orderbook_path, 
        seq_len=seq_len, 
        train_rate=train_rate,
        nrows=None
    )

    # TimeGAN training pipeline
    synthetic_train_data = timegan(train_data, parameters)
    
    # synthetic dataset outputs
    os.makedirs('output', exist_ok=True)
    np.save('output/synthetic_lob_train.npy', synthetic_train_data)
    np.save('output/real_lob_test.npy', np.array(test_data, dtype=object))
    


if __name__ == '__main__':
    main()