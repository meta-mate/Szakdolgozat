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
import matplotlib.pyplot as plt
from visualization import Visualization
from AbstractHRM import AbstractHRM
import arc_modules as am


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
                param_group["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

class Model(nn.Module):
    def __init__(self, d_model=128, option=0):
        super().__init__()
        self.reasoning_block = am.ReasoningBlock(d_model, option=option)
        self.ahrm = AbstractHRM(self.reasoning_block)
        self.grid_embed = am.GridEmbed(d_model)
        self.grid_decode = am.GridDecode(d_model)
        self.combined = None

    def forward(self, train_examples, test_input):
        
        pattern_length = self.ahrm.pattern_reader.pattern_length
        if torch.is_grad_enabled() or pattern_length == 0:
            self.not_started = False
            #train_embedded = self.grid_embed(train_examples)
            test_embedded = self.grid_embed(test_input)
            
            #self.combined = self.grid_combiner(train_examples, test_input)
            B, N, H, W, D = test_embedded.shape
            self.combined = test_embedded.view(B, H, W, D)

        x_lowest = self.combined
        x_greatest = x_lowest

        reasoned = x_lowest
        reasoned = self.ahrm(x_lowest, x_greatest)

        decoded = self.grid_decode(reasoned)

        return decoded

def compete(
        draw_func,
        model1,
        model2,
        optimizer1,
        optimizer2,
        epoch_losses1,
        epoch_losses2,
        tasks,
        batch_size,
        epochs,
        save_each,
        save_name1,
        save_name2,
        ):

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__, torch.cuda.is_available())

    scheduler1 = MultiGroupCosineScheduler(optimizer1, scalar=10, T_0=25, T_mult=1, warmup_steps=5)
    scheduler2 = MultiGroupCosineScheduler(optimizer2, scalar=10, T_0=25, T_mult=1, warmup_steps=5)
    
    keys = list(tasks.keys())
    random.shuffle(keys)
    #tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total parameters: {total_params}")

    max_iterations = 13
    y1 = None
    y2 = None
    
    total_start_time = time.perf_counter()

    already_done = len(epoch_losses1)
    for i_epoch in range(epochs):
        
        test_input = batchable_tasks["test"][:10, 0:1]
        test_output = batchable_tasks["test"][:, 1]
        
        done_amount = 0
        epoch_loss1 = []
        epoch_loss2 = []
        
        while done_amount < len(test_input):

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

                    x_test = torch.randint_like(x_test, 0, 10)
                    
                    random_index = random.randint(0, current_batch_size - 1)

                    for inner_epoch in range(1):
                        start_time = time.perf_counter()
                        for i in range(max_iterations):
                            should_grad1 = True
                            should_grad2 = True
                            if i < 0 or (i < 1 and model1.ahrm.block.option == 0):
                                with torch.no_grad():
                                    y1 = model1(x_train, x_test)
                                should_grad1 = False
                            else:
                                y1 = model1(x_train, x_test)

                            if i < 0 or (i < 1 and model2.ahrm.block.option == 0):
                                with torch.no_grad():
                                    y2 = model2(x_train, x_test)
                                should_grad2 = False
                            else:
                                y2 = model2(x_train, x_test)

                            #x_test = torch.randint_like(x_test, 0, 10)
                            
                            if not should_grad1 and not should_grad2:
                                continue
                            
                            prediction1 = torch.argmax(F.softmax(y1, dim=-1), dim=-1).to(torch.long)
                            prediction2 = torch.argmax(F.softmax(y2, dim=-1), dim=-1).to(torch.long)
                            
                            images = []
                            square_size = 12
                            
                            images.append(Visualization.draw_grid(x_test[random_index][0], square_size))
                            images.append(Visualization.draw_grid(prediction1[random_index], square_size))
                            images.append(Visualization.draw_grid(prediction2[random_index], square_size))
                            '''
                            #latent_value = arc_ahrm.ahrm.pattern_reader.node_list.nodeat(0).value.values.first.value.value
                            latent_value = arc_ahrm.combined
                            with torch.no_grad():
                                y1_latent = arc_ahrm.grid_decode(latent_value)
                                y1_latent = F.softmax(y1_latent, dim=-1)
                                latent_grid = torch.argmax(y1_latent, dim=-1)
                                images.append(Visualization.draw_grid(latent_grid[random_index], square_size))
                            '''
                            
                            horizontal_concat = cv2.hconcat(images)
                            
                            draw_func(horizontal_concat, 'input-target-prediction')
                            
                            loss1 = F.cross_entropy(
                                y1.permute(0, 3, 1, 2),
                                prediction2
                            )

                            loss2 = F.cross_entropy(
                                y2.permute(0, 3, 1, 2),
                                prediction1
                            )

                            print(loss1.item(), loss2.item(), torch.cuda.memory_allocated() / 1024 / 1024)
                            
                            if i == max_iterations - 1:
                                epoch_loss1.append(loss1.item())
                                epoch_loss2.append(loss2.item())
                                print('epoch_loss_avg', np.mean(epoch_loss1), np.mean(epoch_loss2))

                                fig, ax = plt.subplots()
                                mean1 = [np.mean(epoch_loss1[:j+1]) for j in range(len(epoch_loss1))]
                                mean2 = [np.mean(epoch_loss2[:j+1]) for j in range(len(epoch_loss2))]
                                ax.plot(epoch_loss1, color="blue", label=save_name1)
                                ax.plot(epoch_loss2, color="orange", label=save_name2)
                                ax.legend(loc="upper right")
                                fig.canvas.draw()
                                img_rgb_buffer = fig.canvas.tostring_argb()
                                width, height = fig.canvas.get_width_height()
                                plot_image_rgb = np.frombuffer(img_rgb_buffer, dtype=np.uint8).reshape(height, width, 4)[:, :, 1:]
                                plot_image_bgr = cv2.cvtColor(plot_image_rgb, cv2.COLOR_RGB2BGR)
                                plt.close(fig)
                                draw_func(plot_image_bgr, 'loss')

                            if should_grad1:
                                optimizer1.zero_grad()
                                loss1.backward()
                                optimizer1.step()

                            if should_grad2:
                                optimizer2.zero_grad()
                                loss2.backward()
                                optimizer2.step()

                        model1.ahrm.reset()
                        model2.ahrm.reset()
                        end_time = time.perf_counter() - start_time
                        total_time = time.perf_counter() - total_start_time
                        print("end_time", end_time)
                        print("total_time", total_time)

                    done_amount += current_batch_size
                    break
                except torch.cuda.OutOfMemoryError:
                    model1.ahrm.reset()
                    model2.ahrm.reset()
                    batch_size //= 2
                    print(batch_size, "batch_size")
                    if batch_size == 0:
                        break
        
        epoch_losses1.append(np.mean(epoch_loss1))
        #epoch_loss1.clear()
        epoch_losses2.append(np.mean(epoch_loss2))
        #epoch_loss2.clear()

        #scheduler1.step(i_epoch)
        #scheduler2.step(i_epoch)

        keys = list(tasks.keys())
        random.shuffle(keys)
        tasks = {key: tasks[key] for key in keys}

        augmented = LoadDataset.augment(tasks)
        batchable_tasks =  LoadDataset.tasks_to_batchable(augmented)
    
        if (i_epoch + 1) % save_each == 0:

            checkpoint = {
                'epoch': i_epoch,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'epoch_losses': epoch_losses1
            }
        
            #torch.save(checkpoint, script_directory + f"/saved/pt/{save_name1}_{save_name2}_{i_epoch + 1 + already_done}.pth")
            with open(script_directory + f'/saved/json/compete/{save_name1}_learning_{save_name2}_losses.json', 'w') as f:
                json.dump(epoch_loss1, f)

            checkpoint = {
                'epoch': i_epoch,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'epoch_losses': epoch_losses2
            }
        
            #torch.save(checkpoint, script_directory + f"/saved/pt/{save_name2}_{save_name1}_{i_epoch + 1 + already_done}.pth")
            with open(script_directory + f'/saved/json/compete/{save_name2}_learning_{save_name1}_losses.json', 'w') as f:
                json.dump(epoch_loss2, f)
    if batch_size == 0:
        print("batch_size has become 0")

def cv2imshow(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(1)


def start(draw_func=cv2imshow):
    
    d_model = 512
    model1 = Model(d_model, option=0).to("cuda").to(torch.bfloat16)
    model2 = Model(d_model, option=1).to("cuda").to(torch.bfloat16)
    
    base_lr = 1e-3 / 5
    emb_lr = base_lr / 20 #/ (286 - 13)
    rec_lr = base_lr
    dec_lr = emb_lr
    
    optimizer1 = torch.optim.Adam([
        {"params": model1.ahrm.parameters(), "lr": rec_lr},
        {"params": model1.grid_embed.parameters(), "lr": emb_lr},
        #{"params": model1.grid_combiner.parameters(), "lr": emb_lr},
        {"params": model1.grid_decode.parameters(), "lr": dec_lr}
        ])
    
    optimizer2 = torch.optim.Adam([
        {"params": model2.ahrm.parameters(), "lr": rec_lr},
        {"params": model2.grid_embed.parameters(), "lr": emb_lr},
        #{"params": model2.grid_combiner.parameters(), "lr": emb_lr},
        {"params": model2.grid_decode.parameters(), "lr": dec_lr}
        ])

    epoch_losses1 = []
    epoch_losses2 = []

    '''
    checkpoint = torch.load(script_directory + "/saved/pt/arc_ahrm_tet_rec_1e-3_emb_1e-4_6000.pth")
    arc_ahrm.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_losses = checkpoint["epoch_losses"]
    '''

    directories = [
        "/ARC-AGI/data/training",
        #"/ARC-AGI/data/evaluation",
        #"/ARC-AGI-2/data/training",
        #"/ARC-AGI-2/data/evaluation"
    ]

    name1 = model1.ahrm.name
    name2 = model2.ahrm.name


    '''
    for directory in directories:
        name += directory.split("/")[-1][0]

    name += "_rec_" + f"{rec_lr:.0e}" + "_emb_" + f"{emb_lr:.0e}"
    name = name.replace('e+0', 'e+').replace('e-0', 'e-')
    '''
    
    print(name1, name2)

    tasks = {}

    for directory in directories:
        new_tasks = LoadDataset.load_arc_tasks(script_directory + directory)
        tasks.update(new_tasks)
        continue
        for new_key in new_tasks.keys():
            if new_key in tasks:
                del tasks[new_key]
            else:
                tasks[new_key] = new_tasks[new_key]
    
    compete(
        draw_func=draw_func,
        model1=model1,
        model2=model2,
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        epoch_losses1=epoch_losses1,
        epoch_losses2=epoch_losses2,
        tasks=tasks,
        batch_size=1,
        epochs=1,
        save_each=1,
        save_name1=name1,
        save_name2=name2
        )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()