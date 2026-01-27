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
    def __init__(self, optimizer, scalar, T_0, T_mult, warmup_steps):
        self.optimizer = optimizer
        self.scalar = scalar
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_steps = warmup_steps

        self.initial_lrs = []
        for param_group in self.optimizer.param_groups:
            self.initial_lrs.append(param_group["lr"])

    def step(self, t):
        if t > 0 and t % self.T_0 == 0:
            self.T_0 *= self.T_mult
        warmup_steps = self.warmup_steps
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.initial_lrs[i]
            if t < warmup_steps:
                param_group["lr"] = base_lr * float(t) / float(max(1, warmup_steps))
            else:
                min_lr = base_lr / self.scalar
                progress = float(t - warmup_steps) / float(max(1, self.T_0 - warmup_steps))
                progress = float(max(1, progress))
                param_group["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))



def train(
        draw_func,
        arc_ahrm,
        optimizer,
        epoch_losses,
        tasks,
        eval_tasks,
        batch_size,
        epochs,
        save_each,
        save_name="arc_ahrm"
        ):

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__, torch.cuda.is_available())

    warmup_steps = 5
    scheduler = MultiGroupCosineScheduler(optimizer, scalar=5, T_0=100-warmup_steps, T_mult=1, warmup_steps=warmup_steps)
    
    keys = list(tasks.keys())
    random.shuffle(keys)
    tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    max_iterations = 13
    y = None
    
    total_start_time = time.perf_counter()

    eval_epoch_losses = []

    already_done = len(epoch_losses)
    for i_epoch in range(epochs):
        
        test_input = batchable_tasks["test"][:, 0:1]
        test_output = batchable_tasks["test"][:, 1]
        
        done_amount = 0
        epoch_loss = []
        while done_amount < len(batchable_tasks["train"]):

            if batch_size == 0:
                break

            while True:
                try:
                    print(i_epoch + already_done, "epoch", done_amount, "done from", len(batchable_tasks["train"]))
                    '''
                    current_batch_size = batch_size
                    max_size = batch_size * 10
                    max_examples = 0
                    
                    keys = list(tasks.keys())
                    start_index = 0
                    temp_amount = done_amount
                    for i, key in enumerate(keys):
                        temp_amount -= len(tasks[key]["test"])
                        if temp_amount == 0:
                            start_index = i + 1
                            break
                    
                    last_index = 0
                    tasks_length = 0
                    for i in range(start_index, len(keys)):
                        max_examples = max(max_examples, len(tasks[keys[i]]["train"]))
                        tasks_length += len(tasks[keys[i]]["test"])
                        size = (tasks_length) * max_examples
                        last_index = i
                        if size > max_size:
                            last_index -= 1
                            break

                    current_tasks = {keys[i]: tasks[keys[i]] for i in range(start_index, last_index + 1)}
                    
                    current_batch =  LoadDataset.tasks_to_batchable(current_tasks)                    
                    test_input = current_batch["test"][:, 0:1]
                    test_output = current_batch["test"][:, 1]
                    
                    current_batch_size = len(current_batch["train"])
                    '''
                    current_batch_size = min(batch_size, len(batchable_tasks["train"]) - done_amount)
                    start = done_amount
                    end = start + current_batch_size
                    
                    x_train = batchable_tasks["train"][start:end]
                    x_test = test_input[start:end]
                    target = test_output[start:end].to(torch.long)
                    
                    random_index = random.randint(0, current_batch_size - 1)

                    for inner_epoch in range(1):
                        start_time = time.perf_counter()
                        for i in range(max_iterations):
                            if i < max_iterations - 1: #0 or (i < 1 and arc_ahrm.ahrm.block.option == 0):
                                with torch.no_grad():
                                    y = arc_ahrm(x_train, x_test)
                                continue
                            else:
                                y = arc_ahrm(x_train, x_test)
                            
                            prediction = torch.argmax(F.softmax(y, dim=-1), dim=-1)

                            images = []
                            square_size = 12
                            
                            images.append(Visualization.draw_grid(x_test[random_index][0], square_size))
                            images.append(Visualization.draw_grid(target[random_index], square_size))
                            images.append(Visualization.draw_grid(prediction[random_index], square_size))
                            
                            #latent_value = arc_ahrm.ahrm.pattern_reader.node_list.nodeat(0).value.values.first.value.value
                            latent_value = arc_ahrm.combined
                            with torch.no_grad():
                                y1 = arc_ahrm.grid_decode(latent_value)
                                y1 = F.softmax(y1, dim=-1)
                                latent_grid = torch.argmax(y1, dim=-1)
                                images.append(Visualization.draw_grid(latent_grid[random_index], square_size))

                                y1 = arc_ahrm.grid_decode(arc_ahrm.grid_combiner.test_embedded)
                                y1 = F.softmax(y1, dim=-1)
                                latent_grid = torch.argmax(y1, dim=-1)
                                images.append(Visualization.draw_grid(latent_grid[random_index], square_size))
                            horizontal_concat = cv2.hconcat(images)
                            
                            draw_func(horizontal_concat)

                            loss = F.cross_entropy(
                                y.permute(0, 3, 1, 2),
                                target
                            )

                            print(save_name, loss, torch.cuda.memory_allocated() / 1024 / 1024)
                            
                            if i == max_iterations - 1:
                                #draw_func(horizontal_concat)
                                epoch_loss.append(loss.item())
                                print('epoch_loss_avg', np.mean(epoch_loss))

                            if torch.is_grad_enabled():
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                        arc_ahrm.ahrm.reset()
                        end_time = time.perf_counter() - start_time
                        total_time = time.perf_counter() - total_start_time
                        print("end_time", end_time)
                        print("total_time", total_time)

                    done_amount += current_batch_size
                    break
                except torch.cuda.OutOfMemoryError:
                    arc_ahrm.ahrm.reset()
                    batch_size //= 2
                    print(batch_size, "batch_size")
                    if batch_size == 0:
                        break
        
        epoch_losses.append(np.mean(epoch_loss))
        epoch_loss.clear()
        
        #scheduler.step(i_epoch + already_done)

        keys = list(tasks.keys())
        random.shuffle(keys)
        tasks = {key: tasks[key] for key in keys}

        augmented = LoadDataset.augment(tasks)
        batchable_tasks =  LoadDataset.tasks_to_batchable(augmented)
    
        if (i_epoch + 1 + already_done) % save_each == 0:

            checkpoint = {
                'epoch': i_epoch,
                'model_state_dict': arc_ahrm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses
            }

            if eval_tasks is not None:
                torch.save(checkpoint, script_directory + f"/saved/pt/{save_name}_{i_epoch + 1 + already_done}.pth")
            with open(script_directory + f'/saved/json/{save_name}_{i_epoch + 1 + already_done}_epoch_losses.json', 'w') as f:
                json.dump(epoch_losses, f)
    
        if eval_tasks is not None:
            with torch.no_grad():
                train(
                    draw_func=draw_func,
                    arc_ahrm=arc_ahrm,
                    optimizer=optimizer,
                    epoch_losses=eval_epoch_losses,
                    tasks=eval_tasks,
                    eval_tasks=None,
                    batch_size=batch_size,
                    epochs=1,
                    save_each=save_each,
                    save_name="eval_" + save_name
                )
    
    if batch_size == 0:
        print("batch_size has become 0")

def cv2imshow(img):
    cv2.imshow('input-target-prediction', img)
    cv2.waitKey(1)


def start(draw_func):
    
    d_model = 512
    arc_ahrm = ArcAHRM(d_model).to("cuda").to(torch.bfloat16)
    base_lr = 1e-3
    emb_lr = base_lr / 20 #250 #/ (286 - 13)
    rec_lr = base_lr
    dec_lr = emb_lr
    optimizer = torch.optim.Adam([
        {"params": arc_ahrm.ahrm.parameters(), "lr": rec_lr},
        #{"params": arc_ahrm.grid_embed.parameters(), "lr": emb_lr},
        {"params": arc_ahrm.grid_combiner.parameters(), "lr": emb_lr},
        {"params": arc_ahrm.grid_decode.parameters(), "lr": dec_lr}
        ])

    epoch_losses = []
    '''
    checkpoint = torch.load("./saved-new/pt/arc_ahrm_tet_rec_1e-3_emb_1e-4_400.pth")
    arc_ahrm.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_losses = checkpoint["epoch_losses"]
    '''
    directories = [
        #"/ARC-AGI/data/training",
        #"/ARC-AGI/data/evaluation",
        #"/ARC-AGI-2/data/training",
        "/ARC-AGI-2/data/evaluation"
    ]

    name = "arc_" + arc_ahrm.ahrm.name + "_"

    for directory in directories:
        name += directory.split("/")[-1][0]

    name += "_rec_" + f"{rec_lr:.0e}" + "_emb_" + f"{emb_lr:.0e}"
    name = name.replace('e+0', 'e+').replace('e-0', 'e-')

    print(name)

    tasks = {}
    eval_tasks = None #LoadDataset.load_arc_tasks(script_directory + "/ARC-AGI-2/data/evaluation")

    for directory in directories:
        new_tasks = LoadDataset.load_arc_tasks(script_directory + directory)
        tasks.update(new_tasks)
        continue
        for new_key in new_tasks.keys():
            if new_key in tasks:
                del tasks[new_key]
            else:
                tasks[new_key] = new_tasks[new_key]
    
    train(
        draw_func=draw_func,
        arc_ahrm=arc_ahrm,
        optimizer=optimizer,
        epoch_losses=epoch_losses,
        tasks=tasks,
        eval_tasks=eval_tasks,
        batch_size=128-32,
        epochs=3000,
        save_each=25,
        save_name=name
        )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start(cv2imshow)