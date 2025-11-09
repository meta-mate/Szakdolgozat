import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from arc_ahrm import ArcAHRM
from load_dataset import LoadDataset
import os
import random
import time
import json
import cv2
import numpy as np
from visualization import Visualization


script_directory = os.path.dirname(os.path.realpath(__file__))


def train(
        draw_func,
        arc_ahrm,
        optimizer,
        tasks,
        batch_size,
        epochs,
        save_each,
        save_name="arc_ahrm"
        ):

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__, torch.cuda.is_available())

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    keys = list(tasks.keys())
    random.shuffle(keys)
    tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    max_iterations = 13
    y = None

    epoch_losses = []
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
                    print(i_epoch, "epoch", done_amount, "done from", len(batchable_tasks["train"]))
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

        scheduler.step(epoch_loss)

        keys = list(tasks.keys())
        random.shuffle(keys)
        tasks = {key: tasks[key] for key in keys}

        augmented = LoadDataset.augment(tasks)
        batchable_tasks =  LoadDataset.tasks_to_batchable(augmented)
    
        if (i_epoch + 1) % save_each == 0 or i_epoch == 0:
            torch.save(arc_ahrm.state_dict(), script_directory + f"/saved/pt/{save_name}_{i_epoch + 1}.pt")
            with open(script_directory + f'/saved/json/{save_name}_{i_epoch + 1}_epoch_losses.json', 'w') as f:
                json.dump(epoch_losses, f)
    if batch_size == 0:
        print("batch_size has become 0")

def cv2imshow(img):
    cv2.imshow('input-target-prediction', img)
    cv2.waitKey(1)

if __name__ == "__main__":
    
    d_model = 512
    arc_ahrm = ArcAHRM(d_model).to("cuda").to(torch.bfloat16)
    optimizer = torch.optim.Adam(arc_ahrm.parameters(), lr=1e-4)

    tasks = {}
    tasks.update(LoadDataset.load_arc_tasks(script_directory + "/ARC-AGI/data/training"))
    #tasks.update(LoadDataset.load_arc_tasks(script_directory + "/ARC-AGI/data/evaluation"))
    #tasks.update(LoadDataset.load_arc_tasks(script_directory + "/ARC-AGI-2/data/training"))
    #tasks.update(LoadDataset.load_arc_tasks(script_directory + "/ARC-AGI-2/data/evaluation"))

    train(
        draw_func=cv2imshow,
        arc_ahrm=arc_ahrm,
        optimizer=optimizer,
        tasks=tasks,
        batch_size=2,
        epochs=2,
        save_each=2,
        save_name="arc_ahrm_t")
    cv2.destroyAllWindows()