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

    d_model = 512 + 128 + 64
    print("d_model:", d_model)
    arc_ahrm = ArcAHRM(d_model).to("cuda")
    optimizer = torch.optim.Adam(arc_ahrm.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    test_input = batchable_tasks["test"][:, 0]
    test_output = batchable_tasks["test"][:, 1]

    batch_size = 4
    y = None

    for i_epoch in range(4):
        done_amount = 0

        while done_amount < len(batchable_tasks["train"]):

            if batch_size == 0:
                break

            while True:
                try:
                    print(done_amount, "done from", len(batchable_tasks["train"]))
                    batch_size = min(batch_size, len(batchable_tasks["train"]) - done_amount)
                    end = done_amount + batch_size
                    for i in range(13):
                        
                        if i < 2:
                            with torch.no_grad():
                                y = arc_ahrm(batchable_tasks["train"][done_amount:end], test_input[done_amount:end])
                            continue
                        y = arc_ahrm(batchable_tasks["train"][done_amount:end], test_input[done_amount:end])
                        
                        target = test_output[done_amount:end].to(torch.long)
                        num_colors = y.shape[-1]
                        class_weigths = [1.0 for _ in range(num_colors)]
                        class_weigths[0] = .1
                        class_weigths[1] = .2
                        class_weigths = torch.tensor(class_weigths, device=y.device)
                        loss = F.cross_entropy(
                            y.permute(0, 3, 1, 2),
                            target,
                            weight=class_weigths
                            )

                        base_lr = 1e-3

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = base_lr * i * i / 3


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
                    #batch_size = max(batch_size, 1)
                    print(batch_size, "batch_size")
                    if batch_size == 0:
                        break
            
    #y = F.softmax(y, dim=-1)

    #prediction = torch.argmax(y, dim=-1)

    #print(prediction.shape)
    #print(prediction[0])
    #print(arc_ahrm.ahrm.pattern_reader)
    
    if batch_size != 0:
        torch.save(arc_ahrm.state_dict(), "AbstractHRM/saved/arc_ahrm.pt")
    else:
        print("batch_size has become 0")