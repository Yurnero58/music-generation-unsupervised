import torch
import torch.nn as nn
import random

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim)
        # CRITICAL CHANGE: The decoder now takes the PREVIOUS note as input
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = x.size()
        
        # --- ENCODE ---
        _, (h_n, _) = self.encoder(x)
        z = self.fc_latent(h_n[-1])
        
        # --- DECODE ---
        h_0 = self.fc_dec_init(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        outputs = torch.zeros(batch_size, seq_len, 88).to(x.device)
        
        # Start token (a moment of silence to begin the track)
        decoder_input = torch.zeros(batch_size, 1, 88).to(x.device)
        
        h, c = h_0, c_0
        
        # Autoregressive Loop: Predict step-by-step
        for t in range(seq_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = torch.sigmoid(self.fc_out(out))
            outputs[:, t:t+1, :] = pred
            
            # Teacher Forcing: During training, sometimes feed the real next note. 
            # Otherwise, feed the model's own prediction.
            if self.training and random.random() < teacher_forcing_ratio:
                decoder_input = x[:, t:t+1, :] 
            else:
                decoder_input = pred 
                
        return outputs, z