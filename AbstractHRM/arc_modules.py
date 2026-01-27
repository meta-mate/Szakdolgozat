import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ReasoningBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, option=0):
        super().__init__()
        self.d_model = d_model
        
        #self.self_attn_lesser = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        #self.self_attn_greater = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.cross_attn_lesser_to_greater = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        #self.cross_attn_greater_to_lesser = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        #self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.calls = 0
        self.time = 0

        self.option = option

    def forward(self, x_lesser, x_greater, x_previous = None):
        start_time = time.perf_counter()
        
        self.calls += 1
        
        B, H, W, D = x_lesser.shape
        x_lesser = x_lesser.view(B, H * W, D)
        x_greater = x_greater.view(B, -1, D)
        if self.option != 0:
            x_greater = x_greater + x_lesser
            if x_previous is not None:
                x_greater = x_greater + x_previous.view(B, -1, D)
            x_lesser = x_greater
        elif x_previous is not None:
            x_lesser = x_lesser + x_previous.view(B, -1, D)
            x_greater = x_greater + x_previous.view(B, -1, D)
        
        #lesser_sa, _ = self.self_attn_lesser(x_lesser, x_lesser, x_lesser)
        #greater_sa, _ = self.self_attn_greater(x_greater, x_greater, x_greater)

        #x_lesser = self.norm1(x_lesser + self.dropout(lesser_sa))
        #x_greater = self.norm1(x_greater + self.dropout(greater_sa))

        lesser_ca, _ = self.cross_attn_lesser_to_greater(x_lesser, x_greater, x_greater)
        #greater_ca, _ = self.cross_attn_greater_to_lesser(x_greater, x_lesser, x_lesser)

        x_lesser = self.norm2(x_lesser + self.dropout(lesser_ca))
        #x_greater = self.norm2(x_greater + self.dropout(greater_ca))

        #combined = (x_lesser + x_greater) / 2
        combined = x_lesser
        combined = self.norm3(combined + self.dropout(self.ff(combined)))

        combined = combined.view(B, H, W, D)

        self.time += time.perf_counter() - start_time

        #print("reasoning_block", time.perf_counter() - start_time)
        # Reshape back to grid form
        return combined


class GridEmbed(nn.Module):
    def __init__(self, d_model, num_colors=11, num_examples=11, H=30, W=30):
        super().__init__()
        self.d_model = d_model
        
        self.color_embed = nn.Embedding(num_colors, d_model)
        self.row_embed = nn.Embedding(H, d_model)
        self.col_embed = nn.Embedding(W, d_model)
        self.example_embed = nn.Embedding(num_examples, d_model)
        self.role_embed = nn.Embedding(2, d_model)  # 0 = input, 1 = output
        
    def forward(self, grids):
        start_time = time.perf_counter()
        B, N, H, W = grids.shape
        device = grids.device
        D = self.d_model
        
        color_emb = self.color_embed(grids).view(B, N, H, W, D)
        
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
        logits = self.head(seq)
        
        logits = logits.view(B, H, W, -1)
        #print("grid_decode", time.perf_counter() - start_time)
        return logits


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

        self.grid_embed = GridEmbed(d_model)
        self.test_embedded = None

    def forward(self, train_examples, test_input):
        start_time = time.perf_counter()   

        train_embedded = self.grid_embed(train_examples)
        test_embedded = self.grid_embed(test_input)
        
        B, N, H, W, D = train_embedded.shape
        device = train_embedded.device

        self.test_embedded = test_embedded.view(B, H, W, D)

        ones_like = torch.ones_like(train_examples, device=device)
        zeros_like = torch.zeros_like(train_examples, device=device)
        zeros = torch.zeros_like(test_input[0][0], device=device)
        
        condition = []
        for batch in train_examples:
            condition.append([])
            for train_example in batch:
                condition[-1].append(train_example.equal(zeros))

        condition = torch.tensor(condition, device=device)
        key_padding_mask = torch.where(condition.view(B, N, 1, 1), ones_like, zeros_like)
        key_padding_mask = key_padding_mask.bool().view(B, N * H * W)

        context = train_embedded.view(B, N * H * W, D)
        query = test_embedded.view(B, H * W, D)

        attended, _ = self.attn(query, context, context, key_padding_mask=key_padding_mask)
        
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ff(x)))

        x = x.view(B, H, W, D)
        #print("grid_combiner", time.perf_counter() - start_time)

        return x