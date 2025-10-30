import torch
import torch.nn as nn
import torch.nn.functional as F
from AbstractHRM import AbstractHRM
import arc_modules as am
from load_dataset import LoadDataset


class ArcAHRM(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.reasoning_block = am.ReasoningBlock(d_model)
        self.ahrm = AbstractHRM(self.reasoning_block)
        self.grid_embed = am.GridEmbed(d_model)
        self.grid_combiner = am.GridCombiner(d_model)
        self.grid_decode = am.GridDecode(d_model)
        #self.act = am.ACTModule(d_model)
        #self.greatest_shift = am.RoleShift(d_model)

    def forward(self, train_examples, test_examples):
        
        train_embedded = self.grid_embed(train_examples)
        test_embedded = self.grid_embed(test_examples)
        
        combined = self.grid_combiner(train_embedded, test_embedded)
        
        x_lowest = combined
        #x_greatest = self.greatest_shift(x_lowest)
        x_greatest = x_lowest

        reasoned = self.ahrm(x_lowest, x_greatest)

        decoded = self.grid_decode(reasoned)

        return decoded
    