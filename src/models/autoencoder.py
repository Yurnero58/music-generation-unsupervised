import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Encoder: Added num_layers and dropout for higher capacity [cite: 113]
        self.encoder = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc_hidden = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Mirrored capacity [cite: 113]
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # --- ENCODE ---
        _, (h_n, _) = self.encoder(x)
        
        # Isolate the final hidden state from the last LSTM layer
        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1] 
        
        # Create latent vector z (CRITICAL: Removed ReLU to prevent dead space)
        z = self.fc_hidden(last_hidden) 
        
        # --- DECODE ---
        # Map z back to hidden dimension space
        h_d = self.fc_latent(z) 
        
        # Expand h_d to match the num_layers requirement for the decoder's initial state
        h_0 = h_d.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)
        
        # Provide the expanded latent vector as the sequence input for all time steps t
        decoder_input = h_d.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, hidden_dim)
        
        out, _ = self.decoder(decoder_input, (h_0, c_0))
        
        # Sigmoid bounds the output between [0, 1] for the MSE loss comparison
        reconstruction = torch.sigmoid(self.output_layer(out))
        
        return reconstruction, z