import torch
import torch.nn as nn
import math
from tokenizer import Tokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model     # dim of the densve vectors
        self.num_heads = num_heads  
        self.d_k = d_model // num_heads # Dim of each head
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape  #hape: [batch_size, seq_len, d_model]
        
        # Linear Projections
        Q = self.W_q(x) # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split Heads
        # Reshape to (batch, seq, num_heads, d_k) and transpose to (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) #attention scores
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V) # Multiply by value to get context
        
        # Concatenate heads and flatten for the output projection
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context) 
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        # A simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Attention Sublayer
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out) # residual connection and layer norm
        
        # Feed-Forward Sublayer
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class ProteinBERT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=256):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)]) # stack of transformer blocks
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # Pass through each Transformer Block
        for layer in self.layers:
            x = layer(x)
            
        #prediction
        logits = self.fc_out(x)
        return logits
