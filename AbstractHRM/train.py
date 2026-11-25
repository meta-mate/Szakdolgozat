import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
from arc_ahrm import ArcAHRM
from .AbstractHRM import HRMNodeValue
from load_dataset import LoadDataset
import math
import random
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from visualization import Visualization

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


def calculate_accuracy(prediction, target, test_input, for_maze=False):
    misses = torch.where(prediction != target, 1.0, 0.0)
    
    divisor = 1
    if not for_maze:
        color_occurence = [(target == i).sum().item() for i in range(11)]
        amount_present = sum(1 for x in color_occurence if x > 0)
        rarity_score = torch.ones_like(target, dtype=torch.bfloat16)
        for i in range(len(color_occurence)):
            if color_occurence[i] <= 0:
                continue
            rarity_score[target == i] /= color_occurence[i] ** 2
        #misses *= rarity_score
        
        #divisor = rarity_score.sum(dim=(1, 2))
        divisor = torch.where(target != test_input, 1, 0).sum(dim=(1, 2))
        divisor = divisor.clamp(min=1)
    else:
        divisor = torch.where(target == 4, 1, 0).sum(dim=(1, 2))

    result = 1 - (misses.sum(dim=(1, 2)) / divisor).clamp(min=0, max=1).mean()
    return result.item()


def train(
        draw_func,
        arc_ahrm,
        optimizer,
        metrics,
        directories,
        batch_size,
        epochs,
        save_each,
        save_name="arc_ahrm"
        ):

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__, torch.cuda.is_available())

    warmup_steps = 5
    scheduler = MultiGroupCosineScheduler(optimizer, scalar=5, T_0=100-warmup_steps, T_mult=1, warmup_steps=warmup_steps)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    max_iterations = 13
    y = None
    
    total_start_time = time.perf_counter()

    tasks = None

    is_eval = not torch.is_grad_enabled()
    subcategory = "train" if not is_eval else "eval"

    already_done = len(metrics[subcategory]["loss"])
    for i_epoch in range(epochs):

        if arc_ahrm.uses_combiner:
            is_generated = "Generators" in directories["train"][0]

            if is_generated:
                tasks = LoadDataset.generate_tasks(
                    generator_folders=directories["train"],
                    generator_files=[],
                    nr_of_tasks=1,
                    output="AbstractHRM/datasets/",
                )
            elif tasks is None:
                tasks = {}
                for directory in directories["train"]:
                    new_tasks = LoadDataset.load_arc_tasks(script_directory + directory)
                    tasks.update(new_tasks)

            
            augmented = tasks
            if i_epoch > 0:
                keys = list(tasks.keys())
                random.shuffle(keys)
                tasks = {key: tasks[key] for key in keys}
                if not is_generated:
                    augmented = LoadDataset.augment(tasks)
            batchable_tasks =  LoadDataset.tasks_to_batchable(augmented)
        else:
            if tasks is None:
                tasks = []
                for directory in directories["train"]:
                    d = 30 if "maze" in directory else 9
                    tasks += LoadDataset.load_tasks(script_directory + directory, d, d)

            augmented = tasks
            if i_epoch > 0:
                np.random.shuffle(tasks)
                augmented = LoadDataset.shuffle_sudoku(tasks)
            batchable_tasks = torch.from_numpy(augmented).to('cuda')

        batchable_length = 0
        if arc_ahrm.uses_combiner:
            test_input = batchable_tasks["test"][:, 0:1]
            test_output = batchable_tasks["test"][:, 1]
            batchable_length = len(batchable_tasks["train"])
        else:
            test_input = batchable_tasks[:, 0:1]
            test_output = batchable_tasks[:, 1]
            batchable_length = len(batchable_tasks)
        
        done_amount = 0
        losses = []
        accuracies = []
        while done_amount < batchable_length:

            if batch_size == 0:
                break

            while True:
                try:
                    print(i_epoch + already_done, "epoch", done_amount, "done from", batchable_length)
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
                    current_batch_size = min(batch_size, batchable_length - done_amount)
                    start = done_amount
                    end = start + current_batch_size
                    
                    x_train = None
                    if arc_ahrm.uses_combiner:
                        x_train = batchable_tasks["train"][start:end]

                    x_test = test_input[start:end]
                    target = test_output[start:end].to(torch.long)
                    
                    random_index = random.randint(0, current_batch_size - 1)

                    for inner_epoch in range(1):
                        start_time = time.perf_counter()
                        for i in range(max_iterations):
                            if i < max_iterations - 1 and arc_ahrm.ahrm.grad_only_last or i < 1 and arc_ahrm.ahrm.block.option == 0:
                                with torch.no_grad():
                                    y = arc_ahrm(x_train, x_test)
                                continue
                            else:
                                y = arc_ahrm(x_train, x_test)
                            
                            prediction = torch.argmax(F.softmax(y, dim=-1), dim=-1)

                            images = []
                            square_size = 10
                            
                            images.append(Visualization.draw_grid(x_test[random_index][0], square_size))
                            images.append(Visualization.draw_grid(target[random_index], square_size))
                            images.append(Visualization.draw_grid(prediction[random_index], square_size))
                            
                            #latent_value = arc_ahrm.ahrm.pattern_reader.node_list.nodeat(0).value.values.first.value.value
                            latent_value = arc_ahrm.combined
                            if arc_ahrm.uses_combiner:
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

                            if i < max_iterations - 1:
                                loss = loss * (arc_ahrm.ahrm.block.calls / 273)

                            accuracy = calculate_accuracy(prediction, target, x_test)


                            if i == max_iterations - 1:
                                print(save_name, loss, torch.cuda.memory_allocated() / 1024 / 1024)
                                print("accuracy", accuracy)

                            if i == max_iterations - 1:
                                #draw_func(horizontal_concat)
                                losses.append(loss.item())
                                accuracies.append(accuracy)
                                print('losses_avg', np.mean(losses))
                                print("accuracy_avg", np.mean(accuracies))

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
        
        metrics[subcategory]["loss"].append(np.mean(losses))
        metrics[subcategory]["accuracy"].append(np.mean(accuracies))
        losses.clear()
        accuracies.clear()
        
        #scheduler.step(i_epoch + already_done)
    
        if (i_epoch + 1 + already_done) % save_each == 0:

            checkpoint = {
                'epoch': i_epoch,
                'model_state_dict': arc_ahrm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }

            if epochs > 1:
                torch.save(checkpoint, script_directory + f"/saved/pt/{save_name}_{i_epoch + 1 + already_done}.pth")
            with open(script_directory + f'/saved/json/{save_name}_{i_epoch + 1 + already_done}_metrics.json', 'w') as f:
                json.dump(metrics, f)
    
        if directories["eval"]:
            eval_directories = {"train": directories["eval"], "eval": []}

            with torch.no_grad():
                train(
                    draw_func=draw_func,
                    arc_ahrm=arc_ahrm,
                    optimizer=optimizer,
                    metrics=metrics,
                    directories=eval_directories,
                    batch_size=batch_size,
                    epochs=1,
                    save_each=save_each,
                    save_name=save_name
                )
    
    if batch_size == 0:
        print("batch_size has become 0")

def cv2imshow(img):
    cv2.imshow('input-target-prediction', img)
    cv2.waitKey(1)

plt.style.use('dark_background')
def plt_show(img):
    clear_output(wait=False)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def start(draw_func):
    
    d_model = 512
    arc_ahrm = ArcAHRM(d_model, option=2, grad_only_last=False, uses_combiner=True).to("cuda").to(torch.bfloat16)
    base_lr = 1e-3 / 5
    emb_lr = base_lr / 20 #250 #/ (286 - 13)
    rec_lr = base_lr
    dec_lr = emb_lr
    optimizer = torch.optim.Adam([
        {"params": arc_ahrm.ahrm.parameters(), "lr": rec_lr},
        {"params": arc_ahrm.grid_combiner.parameters(), "lr": emb_lr},
        {"params": arc_ahrm.grid_decode.parameters(), "lr": dec_lr}
        ])

    metrics = {
        "train": {"loss": [], "accuracy": []},
        "eval": {"loss": [], "accuracy": []}
        }
    '''
    checkpoint = torch.load("./saved-new/pt/arc_ahrm_tet_rec_1e-3_emb_1e-4_400.pth")
    arc_ahrm.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    metrics = checkpoint["metrics"]
    '''
    directories = {
        "train": [
            #"/ARC-AGI/data/training",
            #"/ARC-AGI/data/evaluation",
            #"/ARC-AGI-2/data/training",
            "/ARC-AGI-2/data/evaluation"
        ],
        "eval": [

        ]
    }

    directories = {
        "train": [
            "Generators/ARC-AGI-1/Train",
            "Generators/ARC-AGI-1/Eval",
            "Generators/ARC-AGI-2/Train"
        ],
        "eval": [
            "Generators/ARC-AGI-2/Eval"
        ]
    }

    name = arc_ahrm.ahrm.name
    name += "_arc_"
    if not arc_ahrm.ahrm.grad_only_last:
        name += "grad_everything_"

    for directory in directories:
        name += directory.split("/")[-1][0]

    name += "_rec_" + f"{rec_lr:.0e}" + "_emb_" + f"{emb_lr:.0e}"
    name = name.replace('e+0', 'e+').replace('e-0', 'e-')

    print(name)

    batch_size = 4
    if not arc_ahrm.ahrm.grad_only_last:
        batch_size = batch_size // 4

    train(
        draw_func=draw_func,
        arc_ahrm=arc_ahrm,
        optimizer=optimizer,
        metrics=metrics,
        directories=directories,
        batch_size=batch_size, #(128-32)//1,
        epochs=3000,
        save_each=25,
        save_name=name
        )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start(plt_show)