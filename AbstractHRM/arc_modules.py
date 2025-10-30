import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Flatten (H, W, d_model) → (H*W, d_model)
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

        # Reshape back to grid form
        return combined.view(B, H, W, D)

class GridEmbed(nn.Module):
    def __init__(self, d_model, max_size=30, num_colors=10):
        super().__init__()
        self.d_model = d_model
        self.color_emb = nn.Embedding(num_colors + 1, d_model)
        self.row_emb   = nn.Embedding(max_size, d_model)
        self.col_emb   = nn.Embedding(max_size, d_model)

    def forward(self, grid):
        device = grid.device
        shape = grid.shape
        H, W = shape[-2:]
        shape += (self.d_model,)
        grid = grid.reshape(-1)
        color = self.color_emb(grid).view(shape)
        rows  = self.row_emb(torch.arange(H, device=device).unsqueeze(1).expand(H, W))
        cols  = self.col_emb(torch.arange(W, device=device).unsqueeze(0).expand(H, W))
        out = color + rows + cols
        return out.view(shape)  # flatten to sequence [N, d]


class GridDecode(nn.Module):
    def __init__(self, d_model, num_colors=10):
        super().__init__()
        self.d_model = d_model
        self.head = nn.Linear(d_model, num_colors + 1)

    def forward(self, seq):
        B, H, W, d_model = seq.shape[-4:]        
        logits = self.head(seq)          # [N, num_colors]
        return logits.view(B, H, W, -1)     # [H, W, num_colors]


class RoleShift(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.d_model)
        y = self.net(x)
        return y.view(shape)
    

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

        self.output_shift = RoleShift(d_model)
        self.example_shift = RoleShift(d_model)

    def forward(self, example_grids, test_input_grid):
        
        B, N, H, W, D = example_grids.shape


        example_grids = self.example_shift(example_grids)
        for i in range(1, N):
            example_grids[:, 2*i:] = self.example_shift(example_grids[:, 2*i:])
        example_grids[:, 1::2] = self.output_shift(example_grids[:, 1::2])
        

        context = example_grids.view(B, N * H * W, D)
        query = test_input_grid.view(B, H * W, D)

        attended, _ = self.attn(query, context, context)
        x1 = self.norm1(query + self.dropout(attended))
        x = self.norm2(x1 + self.dropout(self.ff(x1)))

        return x.view(B, H, W, D)


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