import torch, time

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 4
        d_model = self.d_model
        self.attn = torch.nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.fc1 = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return self.fc1(x)

device = 'cuda'
m = Tiny().to(device)
for b in [4, 8, 16, 32, 64, 128]:
    x = torch.randn(b, m.d_model, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        y = m(x)
    torch.cuda.synchronize()
    print(b, round((time.time()-t0)*1000, 2))
