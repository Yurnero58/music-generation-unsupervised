import torch
import torch.nn as nn

class MusicVAE(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=512, latent_dim=256):
        super(MusicVAE, self).__init__()
        
        # ENCODER: Learns the mean and log-variance of the latent distribution
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # DECODER: Reconstructs the 88-key piano roll from latent sample z
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        # Reparameterization Trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = x.size()
        
        # Encoding phase
        _, (h_n, _) = self.encoder(x)
        h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        mu = self.fc_mu(h_cat)
        logvar = self.fc_logvar(h_cat)
        z = self.reparameterize(mu, logvar)
        
        # Decoding phase
        h = self.fc_dec_init(z).unsqueeze(0)
        c = torch.zeros_like(h)
        
        outputs = torch.zeros(batch_size, seq_len, 88).to(x.device)
        decoder_input = torch.zeros(batch_size, 1, 88).to(x.device)
        
        for t in range(seq_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = torch.sigmoid(self.fc_out(out))
            outputs[:, t:t+1, :] = pred
            
            # Teacher forcing for multi-genre stability
            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = x[:, t:t+1, :]
            else:
                decoder_input = pred
                
        return outputs, mu, logvar