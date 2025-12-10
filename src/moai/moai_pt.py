import torch
import torch.nn as nn

class MoaiTransformerBlockPT(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch, seq, d_model]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MoaiPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = MoaiTransformerBlockPT(
            d_model=config.d_model,
            n_heads=config.n_heads
        )
        
class MoaiConfig:
    def __init__(self, d_model=256, n_heads=8, n_layers=4, max_seq_len=64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
