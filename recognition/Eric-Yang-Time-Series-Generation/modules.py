import torch
import torch.nn as nn
import torch.optim as optim

class Embedder(nn.Module):
    """maps original data space to latent space: e_t = emb(x_t)"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.linear(out))


class Recovery(nn.Module):
    """maps latent space back to data space: x_tilde_t = rec(e_t)"""
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.linear(out)


class Generator(nn.Module):
    """generates synthetic representations in latent space: h_t = gen(h_{t-1}, z_t)"""
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z):
        out, _ = self.rnn(z)
        return torch.sigmoid(self.linear(out))


class Supervisor(nn.Module):
    """captures temporal dynamics in latent space: h_s_t = sup(h_t)"""
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers - 1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.linear(out))


class Discriminator(nn.Module):
    """distinguishes between real latent codes and synthetic codes"""
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.linear(out)


class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, num_layers):
        super(TimeGAN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        
        # initialise sub-networks
        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def train_timegan(self, real_data, epochs=100, batch_size=64, lr=0.001):
        """
        executes the training phases of TimeGAN:
        1) embedding network training (reconstruction)
        2) supervisor training (temporal dynamics)
        3) joint training (generator, discriminator, supervisor)
        """
        # optimise with ADAM
        opt_e = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        opt_s = optim.Adam(self.supervisor.parameters(), lr=lr)
        opt_g = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCEWithLogitsLoss()
        
        dataset_size = real_data.size(0)
        
        print("starting embedding Network Training (Autoencoder)")
        for epoch in range(epochs // 3):
            # simple batch slicing
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                
                # reconstruction Loss
                loss_e = criterion_mse(X_tilde, X) * 10.0
                
                opt_e.zero_grad()
                loss_e.backward()
                opt_e.step()
                
        print("starting supervisor training")
        for epoch in range(epochs // 3):
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                
                H = self.embedder(X)
                H_hat_supervised = self.supervisor(H)
                
                # supervisor Loss (next step prediction in latent space)
                loss_s = criterion_mse(H_hat_supervised[:, :-1, :], H[:, 1:, :])
                
                opt_s.zero_grad()
                loss_s.backward()
                opt_s.step()
                
        print("starting joint training (adversarial + supervised)")
        for epoch in range(epochs // 3):
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                current_batch_size = X.size(0)
                seq_len = X.size(1)
                
                # train generator / supervisor
                Z = torch.randn(current_batch_size, seq_len, self.noise_dim, device=X.device)
                
                H = self.embedder(X)
                E_hat = self.generator(Z)
                H_hat = self.supervisor(E_hat)
                
                Y_fake = self.discriminator(H_hat)
                
                # generator losses
                loss_g_u = criterion_bce(Y_fake, torch.ones_like(Y_fake))
                loss_g_s = criterion_mse(H_hat[:, :-1, :], H[:, 1:, :])
                
                # feature matching loss
                X_hat = self.recovery(H_hat)
                loss_g_v = torch.mean(torch.abs(torch.mean(X_hat, dim=0) - torch.mean(X, dim=0))) + \
                           torch.mean(torch.abs(torch.std(X_hat, dim=0) - torch.std(X, dim=0)))
                
                loss_g = loss_g_u + loss_g_s * 100.0 + loss_g_v * 10.0
                
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                
                # train discriminator
                Z = torch.randn(current_batch_size, seq_len, self.noise_dim, device=X.device)
                E_hat = self.generator(Z)
                H_hat = self.supervisor(E_hat).detach()
                
                Y_real = self.discriminator(H.detach())
                Y_fake = self.discriminator(H_hat)
                
                loss_d_real = criterion_bce(Y_real, torch.ones_like(Y_real))
                loss_d_fake = criterion_bce(Y_fake, torch.zeros_like(Y_fake))
                loss_d = loss_d_real + loss_d_fake
                
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

    @torch.no_grad()
    def generate(self, num_samples, seq_len, device):
        """generates synthetic time-series data with the learned distribution"""
        self.eval()
        Z = torch.randn(num_samples, seq_len, self.noise_dim, device=device)
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        X_hat = self.recovery(H_hat)
        return X_hat

