import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # --- ENCODE ---
        _, (h_n, _) = self.encoder(x)
        z = self.fc_enc(h_n[-1]) # Project final hidden state to latent space
        
        # --- DECODE ---
        # Map z back to hidden dimension to initialize the decoder's brain
        dec_init_h = self.fc_dec(z).unsqueeze(0) 
        dec_init_c = torch.zeros_like(dec_init_h)
        
        # CRITICAL FIX: Feed the expanded latent concept into EVERY timestep
        # This prevents the LSTM from "forgetting" the music halfway through
        dec_inputs = self.fc_dec(z).unsqueeze(1).repeat(1, seq_len, 1)
        
        out, _ = self.decoder(dec_inputs, (dec_init_h, dec_init_c))
        reconstruction = torch.sigmoid(self.out(out))
        
        return reconstruction, z