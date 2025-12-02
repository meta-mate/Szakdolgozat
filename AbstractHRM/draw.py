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

    tasks = LoadDataset.load_arc_tasks("AbstractHRM/ARC-AGI-2/data/training")
    #augmented = LoadDataset.augment(tasks)
    #tasks.update(augmented)
    
    #keys = list(tasks.keys())
    #random.shuffle(keys)
    #tasks = {key: tasks[key] for key in keys}

    batchable_tasks =  LoadDataset.tasks_to_batchable(tasks)

    #d_model = 128 
    d_model = 512
    #d_model = 512 + 128 + 64
    print("d_model:", d_model)
    arc_ahrm = ArcAHRM(d_model).to("cuda").to(torch.bfloat16)
    checkpoint = torch.load("AbstractHRM/saved/pt/arc_ahrm_tet_rec_1e-3_emb_1e-4_6000.pth")
    #arc_ahrm.load_state_dict(torch.load("AbstractHRM/saved/pt/arc_ahrm_tet_reasonlr_1000.pt"))
    arc_ahrm.load_state_dict(checkpoint["model_state_dict"])
    #arc_ahrm.eval()

    total_params = sum(p.numel() for p in arc_ahrm.parameters())
    print(f"Total parameters: {total_params}")

    test_input = batchable_tasks["test"][:, 0:1]
    test_output = batchable_tasks["test"][:, 1]

    batch_index = random.randint(0, len(test_input))
    #batch_index = 225
    print(batch_index)
    
    y = None

    for i in range(13):
        if i < 13 - 1:
            with torch.no_grad():
                y = arc_ahrm(batchable_tasks["train"][batch_index:batch_index + 1], test_input[batch_index:batch_index + 1])
        else:
            y = arc_ahrm(batchable_tasks["train"][batch_index:batch_index + 1], test_input[batch_index:batch_index + 1])

    y = F.softmax(y, dim=-1)
    prediction = torch.argmax(y, dim=-1)

    square_size = 10

    images = []
    images.append(Visualization.draw_grid(test_input[batch_index:batch_index + 1][0][0], square_size))
    images.append(Visualization.draw_grid(test_output[batch_index:batch_index + 1][0], square_size))
    images.append(Visualization.draw_grid(prediction[0], square_size))
    
    i = 0
    j = 0

    while True:
        node_list = arc_ahrm.ahrm.pattern_reader.node_list
        latent_value = node_list.nodeat(i).value.values.nodeat(j).value.value
        #latent_value = arc_ahrm.combined
        
        print(i, j, "Name:", node_list.nodeat(i).value.name)
        
        with torch.no_grad():
            y = arc_ahrm.grid_decode(latent_value)
            y = F.softmax(y, dim=-1)
            latent_grid = torch.argmax(y, dim=-1)
            latent_image = Visualization.draw_grid(latent_grid[0], square_size)

            horizontal_concat = cv2.hconcat(images + [latent_image])

            cv2.imshow('input-target-prediction', horizontal_concat)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        if key == ord("w"):
            if i < len(node_list) - 1:
                i += 1
            j = min(j, len(node_list.nodeat(i).value.values) - 1)
        if key == ord("s"):
            if i > 0:
                i -= 1
            j = min(j, len(node_list.nodeat(i).value.values) - 1)
        if key == ord("d"):
            if j < len(node_list.nodeat(i).value.values) - 1:
                j += 1
        if key == ord("a"):
            if j > 0:
                j -= 1
        

    arc_ahrm.ahrm.reset()
    cv2.destroyAllWindows()