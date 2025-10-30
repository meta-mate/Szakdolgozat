import os
import json
import numpy as np
import random
import torch

class LoadDataset:
    def load_arc_tasks(data_dir):
        tasks = {}
        for fname in os.listdir(data_dir):
            if fname.endswith(".json"):
                with open(os.path.join(data_dir, fname), "r") as f:
                    tasks[fname[:-len(".json")]] = json.load(f)
        return tasks


    def pad_grid(grid, H_max=30, W_max=30, pad_value=-1):
        H, W = len(grid), len(grid[0])
        padded = [[pad_value]*W_max for _ in range(H_max)]
        for r in range(H):
            for c in range(W):
                padded[r][c] = grid[r][c]
        return padded


    def recolor_grid(grid, color_map):
        result = grid.copy()

        for i in range(10):
            result[result == i] += color_map[i] - i

        return result


    def tasks_to_batchable(tasks):
        
        """
        Have the inputs and outputs next to each other.
        Keep the train/test outer dictionary.
        """
        
        result = {"train": [], "test": []}

        max_amount = {"train": 0, "test": 0}
        for task_id in tasks:
            task = tasks[task_id]
            for example_type in result:
                length = len(task[example_type])
                max_amount[example_type] = max(max_amount[example_type], length)

        max_amount["test"] = 4
        
        for task_id in tasks:
            
            task = tasks[task_id]

            for j in range(len(task["test"])):
                result["train"].append([])
                length = max_amount["train"]
                for i in range(length):
                    if i < len(task["train"]):
                        example_io = task["train"][i]
                        for grid in example_io.values():
                            padded_grid = np.array(LoadDataset.pad_grid(grid)) + 1
                            result["train"][-1].append(padded_grid)
                    else:
                        for _ in range(2):
                            empty_grid = np.full((30, 30), 0)
                            result["train"][-1].append(empty_grid)

                result["test"].append([])
                example_io = task["test"][j]
                for grid in example_io.values():
                    padded_grid = np.array(LoadDataset.pad_grid(grid)) + 1
                    result["test"][-1].append(padded_grid)

        for example_type in result:
            new_element = np.array(result[example_type])
            new_element = torch.from_numpy(new_element).to('cuda')
            result[example_type] = new_element

        return result


    def augment(tasks):
        
        augmentations = {}

        for task_id in tasks:
            
            color_map = [i for i in range(2, 10)]
            random.shuffle(color_map)
            color_map = [0, 1] + color_map
            color_map = np.array(color_map)
            rotate_amount = 0
            flip_axis = 2
            
            while rotate_amount == 0 and flip_axis == 2:
                rotate_amount = random.randint(0, 3)
                flip_axis = random.randint(0, 2)

            result_id = task_id
            result_id += "_" + str(rotate_amount) + "_" + str(flip_axis)

            augmentations[result_id] = {}
            augmentations[result_id]["train"] = []
            augmentations[result_id]["test"] = []

            for example_type in augmentations[result_id]:
                #dictionary of "train" or "test"
                
                for example_io in tasks[task_id][example_type]:
                    #array of "input" - "output" dictionaries

                    augmentations[result_id][example_type].append({})
                    
                    #example_io["input"] is already a matrix
                    for example_io_key in example_io:
                        grid = np.array(example_io[example_io_key])
                        grid = LoadDataset.recolor_grid(grid, color_map)

                        if rotate_amount != 0:
                            grid = np.rot90(grid, rotate_amount)
                        if flip_axis != 2:
                            grid = np.flip(grid, flip_axis)

                        augmentations[result_id][example_type][-1][example_io_key] = grid


        return augmentations
