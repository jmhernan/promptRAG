import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Minimal Multi-Head Attention layer.

    Implements the core Transformer attention block:

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    but with:
      - multiple heads (H)
      - learned linear projections for Q, K, V, and output

    Notation:
      B  = batch_size
      L_q = length of query sequence
      L_k = length of key/value sequence
      d_model = model dimension (input/output embedding size)
      H  = num_heads
      d_k = d_model / H (per-head key/query dimension)
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # per-head dimensionality

        # Linear layers that map from model space (d_model) to:
        #   - concatenated Q heads: (d_model → d_model = H * d_k)
        #   - concatenated K heads: (d_model → d_model)
        #   - concatenated V heads: (d_model → d_model)
        #
        # We pack all heads into one big projection and later reshape.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final linear layer to map concatenated heads (H * d_k) back to d_model
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv=None, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            x_q: Tensor of shape (B, L_q, d_model)
                 The sequence whose tokens are asking "queries".
            x_kv: Tensor of shape (B, L_k, d_model) or None
                 The sequence providing "keys" and "values".
                 - Self-attention: x_q == x_kv (same sequence)
                 - Cross-attention: x_q (decoder states), x_kv (encoder outputs)
                 If None, we default to self-attention: x_kv = x_q.
            mask: Optional tensor that is broadcastable to (B, 1, L_q, L_k)
                 Typically:
                   * 0 where attention is allowed
                   * -inf where attention should be blocked
                 The mask is added to the attention scores before softmax.

        Returns:
            output: (B, L_q, d_model)
            attn_weights: (B, H, L_q, L_k)  # attention matrix per head
        """
        # If no separate key/value sequence is provided, use x_q → self-attention
        if x_kv is None:
            x_kv = x_q

        batch_size = x_q.size(0)

        # 1) Project inputs to Q, K, V in model space.
        #
        # Shapes:
        #   x_q : (B, L_q, d_model)
        #   x_kv: (B, L_k, d_model)
        #
        # After linear:
        #   Q, K, V: (B, L_*, d_model)  [still combined heads]
        Q = self.W_q(x_q)  # (B, L_q, d_model)
        K = self.W_k(x_kv) # (B, L_k, d_model)
        V = self.W_v(x_kv) # (B, L_k, d_model)

        # 2) Reshape into separate heads:
        #
        # We want:
        #   Q: (B, H, L_q, d_k)
        #   K: (B, H, L_k, d_k)
        #   V: (B, H, L_k, d_k)
        #
        # Step-by-step:
        #   Q.view(B, L_q, H, d_k) → (B, L_q, H, d_k)
        #   transpose(1, 2) → (B, H, L_q, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now shapes:
        #   Q: (B, H, L_q, d_k)
        #   K: (B, H, L_k, d_k)
        #   V: (B, H, L_k, d_k)

        # 3) Scaled dot-product attention per head.
        #
        # For each head h:
        #   scores[h] = Q[h] K[h]^T / sqrt(d_k)
        #
        # Where:
        #   Q[h]: (B, L_q, d_k)
        #   K[h]: (B, L_k, d_k)
        #   K[h]^T: (B, d_k, L_k)
        #   scores[h]: (B, L_q, L_k)
        #
        # With heads dimension included:
        #   scores: (B, H, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided:
        # Mask is added elementwise to scores:
        #   where mask == -inf, softmax -> 0
        if mask is not None:
            # mask should be broadcastable to scores (B, H, L_q, L_k)
            # Common patterns:
            #   mask: (B, 1, 1, L_k)   for padding
            #   mask: (1, 1, L_q, L_k) for causal masking
            scores = scores + mask

        # 4) Softmax along the key dimension (last dim) to get attention weights.
        #
        # For each query position i, across all keys j:
        #   attn_weights[b, h, i, j] = probability of attending to key j
        # Row sums (over j) ≈ 1.0.
        attn_weights = F.softmax(scores, dim=-1)         # (B, H, L_q, L_k)

        # 5) Use attention weights to combine values.
        #
        #   attn_output = attn_weights @ V
        #
        # Shapes:
        #   attn_weights: (B, H, L_q, L_k)
        #   V          : (B, H, L_k, d_k)
        #   attn_output: (B, H, L_q, d_k)
        attn_output = torch.matmul(attn_weights, V)      # (B, H, L_q, d_k)

        # 6) Concatenate heads back together:
        #
        #   attn_output: (B, H, L_q, d_k)
        #   transpose to (B, L_q, H, d_k)
        #   then reshape to  (B, L_q, H * d_k) = (B, L_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L_q, H, d_k)
        attn_output = attn_output.view(batch_size, -1, self.d_model)  # (B, L_q, d_model)

        # 7) Final linear projection in model space.
        output = self.W_o(attn_output)                   # (B, L_q, d_model)

        # Return:
        #   output: contextualized representations
        #   attn_weights: attention patterns for inspection/visualization
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    Simple 2-layer feed-forward network, applied at each time step independently.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)

    Internally:
      - Expand to d_ff
      - ReLU
      - Project back to d_model
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, d_model)
        x = self.linear1(x)          # (B, L, d_ff)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)          # (B, L, d_model)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer:

      x → LN → MHA → Dropout → +x → x1
      x1 → LN → FFN → Dropout → +x1 → output

    Args:
      d_model   : model dimension
      num_heads : number of attention heads
      d_ff      : hidden size in feed-forward network
      dropout   : dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff=64, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        x: (B, L, d_model)   input embeddings / hidden states
        src_mask: optional attention mask for self-attention
                  broadcastable to (B, 1, L, L)
        """

        # ----- 1) Self-attention sub-layer -----
        # Pre-norm: normalize input first
        x_norm = self.norm1(x)

        # Multi-head self-attention (queries=keys=values=x_norm)
        attn_output, attn_weights = self.self_attn(x_norm, x_norm, mask=src_mask)
        # attn_output: (B, L, d_model)

        # Residual connection + dropout
        x = x + self.dropout1(attn_output)   # (B, L, d_model)

        # ----- 2) Feed-forward sub-layer -----
        x_norm2 = self.norm2(x)

        ffn_output = self.ffn(x_norm2)       # (B, L, d_model)
        x = x + self.dropout2(ffn_output)    # (B, L, d_model)

        # Return:
        #   x: updated hidden states
        #   attn_weights: (B, H, L, L) for visualization/debug
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """
    Mini Transformer encoder = N stacked encoder layers.

    Inputs:
      x: (B, L, d_model)
      src_mask: optional mask (B, 1, L, L) or broadcastable

    Returns:
      x: (B, L, d_model)
      all_attn: list of attention tensors (one per layer)
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        all_attn = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask=src_mask)
            all_attn.append(attn_weights)
        return x, all_attn

# Pretend this is our tokenized sentence
tokens = ["I", "really", "like", "Transformers", "."]

batch_size = 1
seq_len = len(tokens)
d_model = 16
num_heads = 4
num_layers = 2

# Random input embeddings (normally you'd use a real embedding layer)
x = torch.randn(batch_size, seq_len, d_model)

encoder = TransformerEncoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=64,
    dropout=0.0  # set 0.0 for easier debugging
)

# No mask for now (full self-attention allowed)
out, all_attn = encoder(x)

print("Input shape:", x.shape)           # (1, 5, 16)
print("Output shape:", out.shape)        # (1, 5, 16)
print("Num layers:", len(all_attn))      # 2
print("Layer 0 attn shape:", all_attn[0].shape)  # (1, 4, 5, 5)
print("Layer 1 attn shape:", all_attn[1].shape)  # (1, 4, 5, 5)

# Assuming you still have the plotting helpers from earlier:

# Example: layer 0, head 0
plot_multihead_attention(all_attn[0], head_idx=0, tokens=tokens, batch_idx=0)

# Example: layer 1, head 2
plot_multihead_attention(all_attn[1], head_idx=2, tokens=tokens, batch_idx=0)

# ---- Playground / sanity checks ----------------------------------------

d_model = 16
num_heads = 4
batch_size = 2
seq_len = 5

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Random input sequence (e.g., token embeddings)
x = torch.randn(batch_size, seq_len, d_model)

out, attn = mha(x)  # self-attention (x_q == x_kv)

print("Input shape:", x.shape)          # (2, 5, 16)
print("Output shape:", out.shape)       # (2, 5, 16)
print("Attn shape:", attn.shape)        # (2, 4, 5, 5) → (B, H, L_q, L_k)

# Look at head 0 for batch 0
print("Head 0 attention matrix for batch 0:\n", attn[0, 0])
print("Row sums (≈1):", attn[0, 0].sum(-1))  # each row ≈ 1

L = seq_len
# Upper-triangular matrix filled with -inf above the diagonal
# → blocks "future" positions j > i.
mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)
mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L), will broadcast to (B, H, L, L)

out_causal, attn_causal = mha(x, mask=mask)

print("Causal head 0, batch 0:\n", attn_causal[0, 0])
# You should see roughly zero (very dark) above the diagonal after softmax.

def plot_attention_heatmap(attn_matrix, query_tokens, key_tokens, title="Attention"):
    """
    Simple heatmap for a single (L_q, L_k) attention matrix.

    attn_matrix: (L_q, L_k) tensor for one batch and one head
    query_tokens: list of length L_q (e.g., ["I", "love", "Transformers"])
    key_tokens: list of length L_k
    """
    attn_np = attn_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(attn_np)

    # Axis ticks/labels
    ax.set_xticks(np.arange(len(key_tokens)))
    ax.set_yticks(np.arange(len(query_tokens)))
    ax.set_xticklabels(key_tokens, rotation=45, ha="right")
    ax.set_yticklabels(query_tokens)

    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    ax.set_title(title)

    # Colorbar to read intensity
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_multihead_attention(attn, head_idx, tokens, batch_idx=0, title_prefix="Head"):
    """
    Visualize the attention pattern for a specific head and batch element.

    attn: (B, H, L_q, L_k)  from MultiHeadAttention
    head_idx: which head (0 .. H-1) to visualize
    tokens: list of tokens with length L_q == L_k (for self-attention)
    batch_idx: which batch element to visualize (default 0)
    """
    attn_head = attn[batch_idx, head_idx]  # (L_q, L_k)
    title = f"{title_prefix} {head_idx}, batch {batch_idx}"
    plot_attention_heatmap(attn_head, tokens, tokens, title=title)

# Pretend we have this tokenized sentence
tokens = ["I", "really", "like", "Transformers", "."]

seq_len = len(tokens)
batch_size = 1
d_model = 16
num_heads = 4

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Random embeddings for each token (in practice, you'd use learned word embeddings)
X = torch.randn(batch_size, seq_len, d_model)

out, attn = mha(X)  # self-attention

print("out shape:", out.shape)   # (1, L, d_model)
print("attn shape:", attn.shape) # (1, H, L, L)

# Plot each head
for h in range(num_heads):
    plot_multihead_attention(attn, head_idx=h, tokens=tokens, batch_idx=0)
