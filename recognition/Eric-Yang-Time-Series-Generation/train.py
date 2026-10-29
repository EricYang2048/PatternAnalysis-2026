import torch
import time
import os
import matplotlib.pyplot as plt

from modules import TimeGAN
from dataset import get_lobster_timeseries_dataset

def main():
    # hyperparameters
    BATCH_SIZE = 64
    SEQ_LEN = 24
    HIDDEN_DIM = 32
    NOISE_DIM = 32
    NUM_LAYERS = 2
    ITERATIONS = 1000 
    
    # paths for dataset
    orderbook_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
    message_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv'

    # message_path = '/home/Student/s4883721/timeGAN-LOB/amazon_message.csv'
    # orderbook_path = '/home/Student/s4883721/timeGAN-LOB/amazon_orderbook.csv'

    train_rate = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load and preprocess LOBSTER dataset
    print('loading dataset')
    train_data, test_data, train_time, test_time = get_lobster_timeseries_dataset(
        orderbook_path=orderbook_path, 
        seq_len=SEQ_LEN, 
        train_rate=train_rate,
        nrows=50000
    )

    input_dim = train_data.size(-1)
    print(f"dataset loaded train data shape: {train_data.shape}, input dim: {input_dim}")

    # initialise timeGAN model
    model = TimeGAN(
        input_dim=input_dim, 
        hidden_dim=HIDDEN_DIM, 
        noise_dim=NOISE_DIM, 
        num_layers=NUM_LAYERS
    ).to(device)

    # move training tensor to target device
    train_data = train_data.to(device)

    start = time.perf_counter()
    
    # train the model
    print('starting training')
    epochs = max(1, ITERATIONS // 20)
    model.train_timegan(train_data, epochs=epochs, batch_size=BATCH_SIZE)
    
    # generate synthetic time-series data
    print('generating synthetic sequences')
    synthetic_data = model.generate(
        num_samples=40, 
        seq_len=SEQ_LEN, 
        device=device
    )
    
    end = time.perf_counter()
    print(f"Pipeline finished successfully in {end - start:.2f} seconds.")
    print("Synthetic Data Shape:", synthetic_data.shape)

    os.makedirs('output', exist_ok=True)
    torch.save(synthetic_data, 'output/synthetic_lob_train.pt')
    torch.save(test_data, 'output/real_lob_test.pt')
    
    plt.figure(figsize=(12, 5))
    
    # plot spread comparison or cumulative volume
    plt.plot(train_data[0].mean(dim=1).detach().cpu(), label='Real Mean Depth', color='blue', alpha=0.7)
    plt.plot(synthetic_data[0].mean(dim=1).detach().cpu(), label='Synthetic Mean Depth', color='orange', alpha=0.7)
    plt.title('Real vs Synthetic Depth Profile')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Volume')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/train_metrics_comparison.png', dpi=300)
    plt.close()
    print("Training pipeline successfully finished!")

if __name__ == '__main__':
    main()