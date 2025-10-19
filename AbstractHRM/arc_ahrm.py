import torch
import torch.nn as nn
import torch.nn.functional as F
import AbstractHRM
import arc_modules as am
from load_dataset import LoadDataset


class ArcAHRM(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.reasoning_block = am.ReasoningBlock(d_model)
        self.ahrm = AbstractHRM.AbstractHRM(self.reasoning_block)
        self.grid_embed = am.GridEmbed(d_model)
        self.grid_combiner = am.GridCombiner(d_model)
        self.grid_decode = am.GridDecode(d_model)

    def forward(self, train_examples, test_examples):
        
        train_embedded = self.grid_embed(train_examples)
        test_embedded = self.grid_embed(test_examples)
        
        cobmined = self.grid_combiner(train_embedded, test_embedded)
        
        reasoned = self.ahrm(cobmined)

        decoded = self.grid_decode(reasoned)

        return decoded
    

if __name__ == "__main__":

    tasks = LoadDataset.load_arc_tasks("AbstractHRM/ARC/data/evaluation")

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    arc_ahrm = ArcAHRM().to("cuda")

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in arc_ahrm.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    batch_size = 16

    y = None

    while True:
        try:
            print(batch_size)
            y = arc_ahrm(batchable_tasks["train"][:batch_size], batchable_tasks["test"][:batch_size])
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            
    y = F.softmax(y, dim=-1)

    print(y.shape)
    print(y[0])
