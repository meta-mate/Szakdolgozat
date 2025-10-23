import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_ahrm import ArcAHRM
from load_dataset import LoadDataset
import os
import random
import cv2
import numpy as np

def draw_grid(grid, name=""):

    color_list = [
        [0, 0, 0],
        [0, 0, 0],
        [30, 147, 255],
        [249, 60, 49],
        [79, 204, 48],
        [255, 220, 0],
        [153, 153, 153],
        [229, 58, 163],
        [255, 133, 27],
        [135, 216, 241],
        [146, 18, 49]
    ]

    for color in color_list:
        color.reverse()

    y = len(grid)
    x = len(grid[0])

    square_size = 25

    img = np.zeros((y * square_size, x * square_size, 3), dtype=np.uint8)

    for i in range(y):
        for j in range(x):
            start = (j * square_size, i * square_size)
            end = tuple(s + square_size for s in start)
            index = grid[i][j]
            color = color_list[index]
            cv2.rectangle(img, start, end, color, -1)
            if index != 0:
                cv2.rectangle(img, start, end, (127,127,127))

    cv2.imshow(name, img)


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    print(torch.__version__)

    tasks = LoadDataset.load_arc_tasks("AbstractHRM/ARC/data/evaluation")
    #augmented = LoadDataset.augment(tasks)
    #tasks.update(augmented)
    
    #keys = list(tasks.keys())
    #random.shuffle(keys)
    #tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    d_model = 128 #512 + 128 + 64
    print("d_model:", d_model)
    arc_ahrm = ArcAHRM(d_model).to("cuda")
    arc_ahrm.load_state_dict(torch.load("AbstractHRM/saved/arc_ahrm.pt"))
    arc_ahrm.eval()

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    test_input = batchable_tasks["test"][:, 0]
    test_output = batchable_tasks["test"][:, 1]

    batch_index = 3
    
    y = None

    for i in range(13):
        
        if i < 2:
            with torch.no_grad():
                y = arc_ahrm(batchable_tasks["train"][batch_index:batch_index + 1], test_input[batch_index:batch_index + 1])
            continue
        y = arc_ahrm(batchable_tasks["train"][batch_index:batch_index + 1], test_input[batch_index:batch_index + 1])
        
        target = test_output[batch_index:batch_index + 1].to(torch.long)
        loss = F.cross_entropy(y.permute(0, 3, 1, 2), target)

        print(loss)

    arc_ahrm.ahrm.reset()

    y = F.softmax(y, dim=-1)
    prediction = torch.argmax(y, dim=-1)

    draw_grid(test_input[batch_index:batch_index + 1][0], "input")
    draw_grid(prediction[0], "output")
    draw_grid(test_output[batch_index:batch_index + 1][0], "target")

    cv2.waitKey(0)