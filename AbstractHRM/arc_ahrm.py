import torch
import torch.nn as nn
import torch.nn.functional as F
from AbstractHRM import AbstractHRM
import arc_modules as am
from load_dataset import LoadDataset


class ArcAHRM(nn.Module):
    def __init__(self, d_model=128, option=0, uses_combiner=False):
        super().__init__()
        self.reasoning_block = am.ReasoningBlock(d_model, option=option)
        self.ahrm = AbstractHRM(self.reasoning_block)
        self.grid_combiner = am.GridCombiner(d_model)
        self.grid_decode = am.GridDecode(d_model)
        self.combined = None
        self.uses_combiner = uses_combiner

    def forward(self, train_examples, test_input):
        
        pattern_length = self.ahrm.pattern_reader.pattern_length
        if torch.is_grad_enabled() or pattern_length == 0:
            self.not_started = False
            
            if self.uses_combiner:
                self.combined = self.grid_combiner(train_examples, test_input)
            else:
                test_embedded = self.grid_combiner.grid_embed(test_input)
                B, N, H, W, D = test_embedded.shape
                self.combined = test_embedded.view(B, H, W, D)

        x_lowest = self.combined
        x_greatest = x_lowest

        reasoned = x_lowest
        reasoned = self.ahrm(x_lowest, x_greatest)

        decoded = self.grid_decode(reasoned)

        return decoded
    