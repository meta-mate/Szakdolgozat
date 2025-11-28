import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_ahrm import ArcAHRM
from load_dataset import LoadDataset
import os
import math
import random
import time
import json
import cv2
import numpy as np
from visualization import Visualization


script_directory = os.path.dirname(os.path.realpath(__file__))

class MultiGroupCosineScheduler:
    def __init__(self, optimizer, scalar, T_0, T_mult):
        self.optimizer = optimizer
        self.scalar = scalar
        self.T_0 = T_0
        self.T_mult = T_mult

        self.initial_lrs = []
        for param_group in self.optimizer.param_groups:
            self.initial_lrs.append(param_group["lr"])

    def step(self, t):
        if t > 0 and t % self.T_0 == 0:
            self.T_0 *= self.T_mult
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.initial_lrs[i]
            min_lr = base_lr / self.scalar
            T_0 = self.T_0
            param_group["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * (t % T_0) / T_0))


def train(
        draw_func,
        arc_ahrm,
        optimizer,
        epoch_losses,
        tasks,
        batch_size,
        epochs,
        save_each,
        save_name="arc_ahrm"
        ):

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__, torch.cuda.is_available())

    #scheduler = MultiGroupCosineScheduler(optimizer, scalar=2, T_0=50, T_mult=2)
    
    keys = list(tasks.keys())
    random.shuffle(keys)
    tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    max_iterations = 13
    y = None

    already_done = len(epoch_losses)
    for i_epoch in range(epochs):
        
        test_input = batchable_tasks["test"][:, 0:1]
        test_output = batchable_tasks["test"][:, 1]
        
        done_amount = 0
        epoch_loss = 0
        while done_amount < len(batchable_tasks["train"]):

            if batch_size == 0:
                break

            while True:
                try:
                    print(i_epoch + already_done, "epoch", done_amount, "done from", len(batchable_tasks["train"]))
                    current_batch_size = min(batch_size, len(batchable_tasks["train"]) - done_amount)
                    end = done_amount + current_batch_size
                    
                    x_train = batchable_tasks["train"][done_amount:end]
                    x_test = test_input[done_amount:end]
                    target = test_output[done_amount:end].to(torch.long)

                    for inner_epoch in range(1):
                        start_time = time.perf_counter()
                        for i in range(max_iterations):
                            if i < max_iterations - 1:
                                with torch.no_grad():
                                    y = arc_ahrm(x_train, x_test)
                                continue
                            else:
                                y = arc_ahrm(x_train, x_test)
                            
                            prediction = torch.argmax(F.softmax(y, dim=-1), dim=-1)

                            random_index = random.randint(0, current_batch_size - 1)
                            images = []
                            images.append(Visualization.draw_grid(test_input[done_amount + random_index][0]))
                            images.append(Visualization.draw_grid(target[random_index]))
                            images.append(Visualization.draw_grid(prediction[random_index]))

                            #latent_value = arc_ahrm.ahrm.pattern_reader.node_list.nodeat(0).value.values.first.value.value
                            latent_value = arc_ahrm.combined
                            with torch.no_grad():
                                y1 = arc_ahrm.grid_decode(latent_value)
                                y1 = F.softmax(y1, dim=-1)
                                latent_grid = torch.argmax(y1, dim=-1)
                                images.append(Visualization.draw_grid(latent_grid[random_index]))

                            horizontal_concat = cv2.hconcat(images)

                            horizontal_concat = cv2.hconcat(images)
                            
                            draw_func(horizontal_concat)

                            if i < max_iterations - 1:
                                continue

                            loss = F.cross_entropy(
                                y.permute(0, 3, 1, 2),
                                target
                            )

                            epoch_loss += loss.item()

                            print(loss, torch.cuda.memory_allocated() / 1024 / 1024)
                            epoch_loss_avg = epoch_loss / ((done_amount) / batch_size + 1)
                            print('epoch_loss_avg', epoch_loss_avg)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        arc_ahrm.ahrm.reset()
                        end_time = time.perf_counter() - start_time
                        print("end_time", end_time)

                    done_amount += current_batch_size
                    break
                except torch.cuda.OutOfMemoryError:
                    arc_ahrm.ahrm.reset()
                    batch_size //= 2
                    print(batch_size, "batch_size")
                    if batch_size == 0:
                        break
        
        epoch_loss /= len(test_input) / batch_size
        epoch_losses.append(epoch_loss)

        #scheduler.step(i_epoch)

        keys = list(tasks.keys())
        random.shuffle(keys)
        tasks = {key: tasks[key] for key in keys}

        augmented = LoadDataset.augment(tasks)
        batchable_tasks =  LoadDataset.tasks_to_batchable(augmented)
    
        if (i_epoch + 1) % save_each == 0:

            checkpoint = {
                'epoch': i_epoch,
                'model_state_dict': arc_ahrm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses
            }
        
            torch.save(checkpoint, script_directory + f"/saved/pt/{save_name}_{i_epoch + 1 + already_done}.pth")
            with open(script_directory + f'/saved/json/{save_name}_{i_epoch + 1 + already_done}_epoch_losses.json', 'w') as f:
                json.dump(epoch_losses, f)
    if batch_size == 0:
        print("batch_size has become 0")

def cv2imshow(img):
    cv2.imshow('input-target-prediction', img)
    cv2.waitKey(1)

if __name__ == "__main__":
    
    d_model = 512
    arc_ahrm = ArcAHRM(d_model).to("cuda").to(torch.bfloat16)
    base_lr = 1e-3
    emb_lr = base_lr / 10 #/ (286 - 13)
    rec_lr = base_lr
    decoder_lr = emb_lr
    optimizer = torch.optim.Adam([
        {"params": arc_ahrm.ahrm.parameters(), "lr": rec_lr},
        {"params": arc_ahrm.grid_embed.parameters(), "lr": emb_lr},
        {"params": arc_ahrm.grid_combiner.parameters(), "lr": emb_lr},
        {"params": arc_ahrm.grid_decode.parameters(), "lr": decoder_lr}
        ])

    directories = [
        "/ARC-AGI/data/training",
        "/ARC-AGI/data/evaluation",
        "/ARC-AGI-2/data/training",
        #"/ARC-AGI-2/data/evaluation"
    ]

    epoch_losses = []
    '''
    checkpoint = torch.load(script_directory + "/saved/pt/arc_ahrm_tet_reasonlr_2350.pth")
    arc_ahrm.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_losses = checkpoint["epoch_losses"]
    '''
    
    tasks = {}

    for directory in directories:
        new_tasks = LoadDataset.load_arc_tasks(script_directory + directory)
        tasks.update(new_tasks)

    train(
        draw_func=cv2imshow,
        arc_ahrm=arc_ahrm,
        optimizer=optimizer,
        epoch_losses=epoch_losses,
        tasks=tasks,
        batch_size=2,
        epochs=2,
        save_each=2,
        save_name="arc_ahrm_tet_slc")
    cv2.destroyAllWindows()