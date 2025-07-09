import torch
import torch.nn as nn

class MyModel(nn.Module):
    """
    Simple LSTM-based model for text processing.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # LSTM layer expects (batch, seq, features)
        # For character-level: features=1, seq=input_size
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        # Reshape to (batch_size, sequence_length, 1) for LSTM
        x = x.unsqueeze(-1)
        
        # Initialize hidden states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        
        # Apply dropout and final linear layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
