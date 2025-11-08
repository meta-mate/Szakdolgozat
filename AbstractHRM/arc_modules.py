import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ReasoningBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Self-attention for each grid
        self.self_attn_lesser = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_greater = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention between grids
        self.cross_attn_lesser_to_greater = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_greater_to_lesser = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feed-forward layer (shared or separate, here we use shared for symmetry)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_lesser, x_greater):
        start_time = time.perf_counter()
        # Flatten (H, W, d_model) to (H*W, d_model)
        B, H, W, D = x_lesser.shape
        x_lesser = x_lesser.view(B, H * W, D)
        x_greater = x_greater.view(B, H * W, D)

        # --- Step 1: Self-attention within each grid ---
        lesser_sa, _ = self.self_attn_lesser(x_lesser, x_lesser, x_lesser)
        greater_sa, _ = self.self_attn_greater(x_greater, x_greater, x_greater)

        x_lesser = self.norm1(x_lesser + self.dropout(lesser_sa))
        x_greater = self.norm1(x_greater + self.dropout(greater_sa))

        # --- Step 2: Cross-attention between grids ---
        lesser_ca, _ = self.cross_attn_lesser_to_greater(x_lesser, x_greater, x_greater)
        greater_ca, _ = self.cross_attn_greater_to_lesser(x_greater, x_lesser, x_lesser)

        x_lesser = self.norm2(x_lesser + self.dropout(lesser_ca))
        x_greater = self.norm2(x_greater + self.dropout(greater_ca))

        # --- Step 3: Feed-forward refinement ---
        combined = (x_lesser + x_greater) / 2  # merge both perspectives
        combined = self.norm3(combined + self.dropout(self.ff(combined)))

        combined = combined.view(B, H, W, D)

        #print("reasoning_block", time.perf_counter() - start_time)
        # Reshape back to grid form
        return combined


class GridEmbed(nn.Module):
    def __init__(self, d_model, num_colors=11, num_examples=11, H=30, W=30):
        """
        num_colors: total number of possible grid values (including 'empty')
        num_examples: number of input/output pairs per sample
        d_model: dimension of the embeddings
        H, W: grid dimensions
        """
        super().__init__()
        self.d_model = d_model
        
        # Embeddings for each grid dimension / role
        self.color_embed = nn.Embedding(num_colors, d_model)
        self.row_embed = nn.Embedding(H, d_model)
        self.col_embed = nn.Embedding(W, d_model)
        self.example_embed = nn.Embedding(num_examples, d_model)
        self.role_embed = nn.Embedding(2, d_model)  # 0 = input, 1 = output
        
        # Final projection to combine all sources
        #self.proj = nn.Linear(d_model * 5, d_model)
    
    def forward(self, grids):
        """
        grids: Tensor (B, N, H, W) of integer color indices
        where:
          - B: batch size
          - N: number of grids (alternating input/output)
        """
        start_time = time.perf_counter()
        B, N, H, W = grids.shape
        device = grids.device
        D = self.d_model
        
        # Base color embeddings
        color_emb = self.color_embed(grids).view(B, N, H, W, D)
        
        # Spatial encodings
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        row_emb = self.row_embed(rows).view(1, 1, H, 1, D)
        col_emb = self.col_embed(cols).view(1, 1, 1, W, D)
        
        # Example (pair) encodings: [0,0,1,1,2,2,...]
        example_ids = torch.tensor([0], device=device)
        if N > 1:
            example_ids = torch.arange(N, device=device) // 2 + 1
        example_emb = self.example_embed(example_ids).view(1, N, 1, 1, D)
        
        # Role encodings: [0,1,0,1,0,1,...]
        role_ids = torch.tensor([0], device=device)
        if N > 1:
            role_ids = torch.arange(N, device=device) % 2
        role_emb = self.role_embed(role_ids).view(1, N, 1, 1, D)
        
        emb = (
            color_emb +
            row_emb.view(1, 1, H, 1, D) +
            col_emb.view(1, 1, 1, W, D) +
            example_emb.view(1, N, 1, 1, D) +
            role_emb.view(1, N, 1, 1, D)
        )

        #print("grid_embed", time.perf_counter() - start_time, grids.shape)
        
        return emb  # shape: (B, N, H, W, d_model)


class GridDecode(nn.Module):
    def __init__(self, d_model, num_colors=10):
        super().__init__()
        self.d_model = d_model
        self.head = nn.Linear(d_model, num_colors + 1)

    def forward(self, seq):
        start_time = time.perf_counter()
        B, H, W, d_model = seq.shape[-4:]        
        logits = self.head(seq)          # [N, num_colors]
        
        logits = logits.view(B, H, W, -1)
        #print("grid_decode", time.perf_counter() - start_time)
        return logits     # [H, W, num_colors]


class GridCombiner(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, example_grids, test_input_grid):
        start_time = time.perf_counter()   
        B, N, H, W, D = example_grids.shape

        context = example_grids.view(B, N * H * W, D)
        #context = example_grids.mean(dim=1).view(B, H * W, D)
        query = test_input_grid.view(B, H * W, D)

        attended, _ = self.attn(query, context, context)
        
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ff(x)))

        x = x.view(B, H, W, D)
        #print("grid_combiner", time.perf_counter() - start_time)

        return x


class ACTModule(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, grid):
        # grid: (B, H, W, D)
        B, H, W, D = grid.shape
        seq = grid.view(B, H * W, D)
        # Use the mean as the query token
        query = seq.mean(dim=1, keepdim=True)
        attended, _ = self.attn(query, seq, seq)
        normalized = self.norm(attended.squeeze(1))  # (B, D)
        return self.fc(normalized) # (B, 1)