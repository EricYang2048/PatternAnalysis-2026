import numpy as np
import pandas as pd
from utils import extract_time

def load_lobster_csv(message_path: str, orderbook_path: str, nrows: int = None):    
    obs = pd.read_csv(orderbook_path, header=None, nrows=nrows)
    msgs = pd.read_csv(message_path, header=None, nrows=nrows)
    
    # message file has 6 columns: [Time, Type, Order ID, Size, Price, Direction]
    # orderbook file has 40 columns for Level 10
    # we combine message timestamp/type with the order book depth features
    combined = pd.concat([msgs.iloc[:, :2], obs], axis=1)
    return combined.values

def preprocess_lob_data(raw_data: np.ndarray):
    data = raw_data.copy()
    
    # orderbook columns start at index 2
    ob_cols = data[:, 2:]
    
    ask_prices_list = []
    bid_prices_list = []
    ask_volumes_list = []
    bid_volumes_list = []
    
    for i in range(10):
        # 4 columns per level: index 0 (Ask Price), index 1 (Ask Size), index 2 (Bid Price), index 3 (Bid Size)
        base_idx = i * 4
        
        # extract and scale prices from LOBSTER format (divide by 10000)
        a_price = ob_cols[:, base_idx] / 10000.0
        a_size  = ob_cols[:, base_idx + 1]
        b_price = ob_cols[:, base_idx + 2] / 10000.0
        b_size  = ob_cols[:, base_idx + 3]
        
        # handle unoccupied price levels (LOBSTER dummy values: 9999999999 / -9999999999)
        a_price = np.where(a_price > 1e6, np.nan, a_price)
        b_price = np.where(b_price < -1e6, np.nan, b_price)
        
        ask_prices_list.append(a_price)
        bid_prices_list.append(b_price)
        ask_volumes_list.append(a_size)
        bid_volumes_list.append(b_size)
        
    ask_prices = np.column_stack(ask_prices_list)
    bid_prices = np.column_stack(bid_prices_list)
    ask_volumes = np.column_stack(ask_volumes_list)
    bid_volumes = np.column_stack(bid_volumes_list)
    
    # fill any NaN values resulting from unoccupied levels forward or with zeros
    ask_prices = pd.DataFrame(ask_prices).ffill().bfill().values
    bid_prices = pd.DataFrame(bid_prices).ffill().bfill().values
    
    # compute log-returns for Level 1 best ask and bid prices
    best_ask_returns = np.diff(np.log(np.maximum(ask_prices[:, 0], 1e-8)))
    best_ask_returns = np.hstack([[0.0], best_ask_returns])
    
    best_bid_returns = np.diff(np.log(np.maximum(bid_prices[:, 0], 1e-8)))
    best_bid_returns = np.hstack([[0.0], best_bid_returns])
    
    # mormalise volumes across all 10 levels using z-score standardisation
    all_volumes = np.hstack([ask_volumes, bid_volumes])
    vol_mean = np.mean(all_volumes, axis=0)
    vol_std = np.std(all_volumes, axis=0) + 1e-8
    norm_volumes = (all_volumes - vol_mean) / vol_std
    
    # combine returns and normalized volumes into final feature matrix
    processed_data = np.column_stack([best_ask_returns, best_bid_returns, norm_volumes])
    return processed_data

def create_sliding_windows(data: np.ndarray, seq_len: int = 24, step: int = 1):
    sequences = []
    total_len = len(data)
    
    for i in range(0, total_len - seq_len + 1, step):
        seq = data[i : i + seq_len, :]
        sequences.append(seq)
        
    return sequences

def chronological_train_test_split(sequences: list, train_rate: float = 0.8):
    split_idx = int(len(sequences) * train_rate)
    
    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]
    
    return train_data, test_data

def get_lobster_timeseries_dataset(message_path: str, orderbook_path: str, seq_len: int = 24, train_rate: float = 0.8, nrows: int = None):
    raw_data = load_lobster_csv(message_path, orderbook_path, nrows=nrows)
    processed_data = preprocess_lob_data(raw_data)
    sequences = create_sliding_windows(processed_data, seq_len=seq_len)
    
    train_data, test_data = chronological_train_test_split(sequences, train_rate=train_rate)
    
    train_time, _ = extract_time(train_data)
    test_time, _ = extract_time(test_data)
    
    return train_data, test_data, train_time, test_time