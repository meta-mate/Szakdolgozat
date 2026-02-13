import os
import json
import csv
import numpy as np
import random
import torch

class LoadDataset:

    def load_sudoku(path):
        with open(path, "r") as f:
            csv_reader = csv.reader(f)
            tasks = list(csv_reader)
        
        result = []
        for i_task, task in enumerate(tasks):
            if i_task == 0:
                continue
            
            result.append([])
            for io in range(2):
                result[-1].append([])
                task_str = task[1 + io]
                task_str = task_str.replace(".", "0")
                for i in range(9):
                    result[-1][-1].append([])
                    for j in range(9):
                        result[-1][-1][-1].append(int(task_str[i * 9 + j]))

        result = np.array(result)
        return result
    
    def shuffle_sudoku(tasks):
        
        for i in range(len(tasks)):
            digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
            
            transpose_flag = np.random.rand() < 0.5

            bands = np.random.permutation(3)
            row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

            stacks = np.random.permutation(3)
            col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

            mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

            def apply_transformation(x):
                if transpose_flag:
                    x = x.T
                new_board = x.flatten()[mapping].reshape(9, 9).copy()
                return digit_map[new_board]
            
            for j in range(2):
                tasks[i][j] = apply_transformation(tasks[i][j])
                continue

        return tasks

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
            result[grid == i] += color_map[i] - i

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
            
            color_map = [i for i in range(1, 10)]
            random.shuffle(color_map)
            color_map = [0] + color_map
            color_map = np.array(color_map)
            
            rotate_amount = random.randint(0, 3)
            flip_axis = random.randint(0, 2)

            result_id = task_id
            result_id += "_" + str(rotate_amount) + "_" + str(flip_axis) + "_"

            for i in range(1, len(color_map)):
                result_id += str(color_map[i])

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
