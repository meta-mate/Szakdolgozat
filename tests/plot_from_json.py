import json
import matplotlib.pyplot as plt
import numpy as np
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
print(script_directory)

per_epoch_updates = 1270 // 96

base_lr = 1e-4
lr = base_lr
weight_decay = 5e-3
values = []

for i in range(100):
    values.append(lr)
    for j in range(per_epoch_updates):
        lr -= lr * weight_decay

values = np.array(values) / base_lr

#plt.plot(values, color="red")

files = {
    "Transformer_arc_tet_rec_1e-3_emb_5e-5_25": {"color": "blue", "time": 1, "label": "Transformer"},
    "AbstractHRM_arc_tet_rec_1e-3_emb_5e-5_25": {"color": "orange", "time": 1, "label": "AbstractHRM"},
    "TRM_arc_tet_rec_1e-3_emb_5e-5_25": {"color": "green", "time": 1, "label": "TRM"}
}

name1 = "Transformer"
name2 = "TRM"

files = {
    f"compete/{name1}_learning_{name2}": {"color": "blue", "time": 1, "label": "Transformer"},
    f"compete/{name2}_learning_{name1}": {"color": "orange", "time": 1, "label": "AbstractHRM"},
}

metric = "loss"
metric = "accuracy"

for file in files:
    datas = []
    with open(f'./AbstractHRM/saved/json/{file}_losses.json', "r") as f:
        datas = np.array(json.load(f))
    x_values = np.arange(len(datas)) * files[file]["time"]
    mean = [datas[:i+1].mean() for i in range(len(datas))]
    label = file.split("_")[0]
    plt.plot(x_values, datas, color=files[file]["color"], label=label)

if metric == "loss":    
    plt.legend(loc="upper right")
if metric == "accuracy":    
    plt.legend(loc="lower right")
plt.show()
