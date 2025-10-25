import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_ahrm import ArcAHRM
from load_dataset import LoadDataset
import os
import random
import cv2
import numpy as np
from visualization import Visualization

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

    d_model = 128 
    #d_model = 512 + 128 + 64
    print("d_model:", d_model)
    arc_ahrm = ArcAHRM(d_model).to("cuda")
    arc_ahrm.load_state_dict(torch.load("AbstractHRM/saved/arc_ahrm.pt"))
    arc_ahrm.eval()

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    test_input = batchable_tasks["test"][:, 0]
    test_output = batchable_tasks["test"][:, 1]

    batch_index = 64
    
    y = None

    for i in range(13):
        
        if i < 1:
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

    images = []
    images.append(Visualization.draw_grid(test_input[batch_index:batch_index + 1][0]))
    images.append(Visualization.draw_grid(test_output[batch_index:batch_index + 1][0]))
    images.append(Visualization.draw_grid(prediction[0]))

    horizontal_concat = cv2.hconcat(images)
    cv2.imshow('input-target-prediction', horizontal_concat)

    cv2.waitKey(0)