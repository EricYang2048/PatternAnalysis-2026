# dataset.py
import csv
import torch

def load_lobster_csv(message_path: str, orderbook_path: str, nrows: int = None) -> torch.Tensor:
    """loads message and orderbook CSV files and combines them"""
    obs_rows = []
    with open(orderbook_path, mode='r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if nrows is not None and i >= nrows:
                break
            float_row = [float(val) for val in row if val.strip() != '']
            if float_row:
                obs_rows.append(float_row)

    msgs_rows = []
    with open(message_path, mode='r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if nrows is not None and i >= nrows:
                break
            float_row = [float(val) for val in row[:2] if val.strip() != '']
            if float_row:
                msgs_rows.append(float_row)

    min_len = min(len(obs_rows), len(msgs_rows))
    obs_tensor = torch.tensor(obs_rows[:min_len], dtype=torch.float32)
    msgs_tensor = torch.tensor(msgs_rows[:min_len], dtype=torch.float32)
    
    return torch.cat([msgs_tensor, obs_tensor], dim=1)

def preprocess_lob_data(raw_data: torch.Tensor) -> torch.Tensor:
    """normalises limit order book data"""
    data_min = raw_data.min(dim=0, keepdim=True).values
    data_max = raw_data.max(dim=0, keepdim=True).values
    processed_data = (raw_data - data_min) / (data_max - data_min + 1e-8)
    return processed_data

def create_sliding_windows(data: torch.Tensor, seq_len: int = 20, step: int = 1) -> torch.Tensor:
    """creates sliding windows"""
    sequences = []
    total_len = data.size(0)
    if total_len < seq_len:
        return torch.empty(0, seq_len, data.size(1))
    for i in range(0, total_len - seq_len + 1, step):
        sequences.append(data[i : i + seq_len])
    return torch.stack(sequences) if sequences else torch.empty(0)

def extract_time_tensor(data_tensor: torch.Tensor):
    """returns valid time tensors for sliding windows"""
    batch_size = data_tensor.size(0)
    seq_len = data_tensor.size(1)
    time_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32)
    return time_tensor, seq_len

def get_lobster_timeseries_dataset(message_path: str, orderbook_path: str, seq_len: int = 20, train_rate: float = 0.8, nrows: int = None):
    """loads and preprocesses the LOBSTER dataset"""
    raw_data = load_lobster_csv(message_path, orderbook_path, nrows=nrows)
    processed_data = preprocess_lob_data(raw_data)
    sequences_tensor = create_sliding_windows(processed_data, seq_len=seq_len)

    split_idx = int(sequences_tensor.size(0) * train_rate)
    train_data = sequences_tensor[:split_idx]
    test_data = sequences_tensor[split_idx:]
    
    train_time, _ = extract_time_tensor(train_data)
    test_time, _ = extract_time_tensor(test_data)
    
    return train_data, test_data, train_time, test_time