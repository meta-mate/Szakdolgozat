import json
import matplotlib.pyplot as plt
import numpy as np

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
    "arc_Transformer_e_rec_1e-3_emb_5e-5_100_epoch_losses": {"color": "blue", "time": 1, "label": "Transformer"},
    "arc_AbstractHRM_e_rec_1e-3_emb_5e-5_100_epoch_losses": {"color": "orange", "time": 1, "label": "AbstractHRM"},
    "arc_TRM_e_rec_1e-3_emb_5e-5_100_epoch_losses": {"color": "green", "time": 1, "label": "TRM"}
}

name1 = "Transformer"
name2 = "TRM_regular"

files_ = {
    f"{name1}_learning_{name2}_losses": {"color": "blue", "time": 1, "label": "AbstractHRM"},
    f"{name2}_learning_{name1}_losses": {"color": "orange", "time": 1, "label": "Transformer"},
}

for file in files:
    loss_datas = []
    with open(f'./AbstractHRM/saved/json/{file}.json', "r") as f:
        loss_datas = np.array(json.load(f))
    x_values = np.arange(len(loss_datas)) * files[file]["time"]
    mean = [loss_datas[:i+1].mean() for i in range(len(loss_datas))]
    label = files[file]["label"]
    #label = file.split("_")[0]
    plt.plot(x_values, loss_datas, color=files[file]["color"], label=label)

plt.legend(loc="upper right")
plt.show()
