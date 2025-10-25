import torch
import torch.nn.functional as F

class CustomFunctions:
    def cross_entropy_per_class(logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) with class indices
        returns: (B, H, W, C) – per-class cross entropy
        """
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        one_hot = F.one_hot(target, num_classes=logits.shape[1])  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # -> (B, C, H, W)
        
        # Elementwise negative log likelihood, *without* reducing over classes
        loss_per_class = -one_hot * log_probs  # (B, C, H, W)
        loss_per_class = loss_per_class.permute(0, 2, 3, 1)  # -> (B, H, W, C)

        B, C, H, W = logits.shape
                        
        src = torch.ones(B, H * W, dtype=target.dtype, device=target.device)
        counts = torch.zeros(B, H * W, dtype=target.dtype, device=target.device)
        counts.scatter_add_(-1, target.view(B, -1), src)
        counts = counts[:, :C]
        counts = torch.where(counts == 0, counts, H * W / 2)

        freq = counts / H / W
        inv_sqrt = 1.0 / torch.sqrt(freq + 1e-6)
        weights = inv_sqrt / torch.mean(inv_sqrt, -1).unsqueeze(1)
        weights = torch.clamp(weights, 0.25, 2)
        weights = weights / torch.mean(weights, -1).unsqueeze(1)

        weighted_loss = loss_per_class * weights[:, None, None, :]

        loss = weighted_loss.sum(dim=-1).mean()
        
        return loss