import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_ahrm import ArcAHRM
from load_dataset import LoadDataset
import os
import random
    

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__)

    tasks = LoadDataset.load_arc_tasks("AbstractHRM/ARC/data/training")
    augmented = LoadDataset.augment(tasks)
    
    tasks.update(augmented)
    keys = list(tasks.keys())
    random.shuffle(keys)
    tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    arc_ahrm = ArcAHRM().to("cuda")
    optimizer = torch.optim.Adam(arc_ahrm.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in arc_ahrm.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    test_input = batchable_tasks["test"][:, 0]
    test_output = batchable_tasks["test"][:, 1]

    batch_size = 8
    done_amount = 0

    while done_amount < len(batchable_tasks["train"]):

        y = None

        while True:
            try:
                print(done_amount, "done from", len(batchable_tasks["train"]))
                batch_size = min(batch_size, len(batchable_tasks["train"]) - done_amount)
                end = done_amount + batch_size
                for i in range(5):
                    
                    if i < 2:
                        with torch.no_grad():
                            y = arc_ahrm(batchable_tasks["train"][done_amount:end], test_input[done_amount:end])
                        continue
                    y = arc_ahrm(batchable_tasks["train"][done_amount:end], test_input[done_amount:end])
                    
                    target = test_output[done_amount:end].to(torch.long)
                    loss = F.cross_entropy(y.permute(0, 3, 1, 2), target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(loss)

                arc_ahrm.ahrm.reset()

                done_amount += batch_size
                break
            except torch.cuda.OutOfMemoryError:
                arc_ahrm.ahrm.reset()
                batch_size //= 2
                batch_size = max(batch_size, 1)
                print(batch_size, "batch_size")
            
    #y = F.softmax(y, dim=-1)

    #prediction = torch.argmax(y, dim=-1)

    #print(prediction.shape)
    #print(prediction[0])
    #print(arc_ahrm.ahrm.pattern_reader)
    
    torch.save(arc_ahrm.state_dict(), "AbstractHRM/saved/arc_ahrm.pt")