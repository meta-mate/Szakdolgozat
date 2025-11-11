import json
import matplotlib.pyplot as plt
import numpy as np


per_epoch_updates = 1270 // 96

base_lr = 1e-4
lr = base_lr
weight_decay = 0.0025
values = []

for i in range(100):
    values.append(lr)
    for j in range(per_epoch_updates):
        lr -= lr * weight_decay

values = np.array(values) / base_lr

#plt.plot(values, color="red")

file_names = {
    "arc_ahrm_tet_reasonlr_150_epoch_losses": "pink",
    "arc_ahrm_tete_cosine_400_epoch_losses": "blue",
    #"arc_ahrm_tete_lr1e-4_24batch_100_epoch_losses": "pink",
    #"arc_ahrm_tete_lr1e-4_100_epoch_losses": "purple",
    #"arc_ahrm_tete_50_4e-3_epoch_losses": "blue",
    #"arc_ahrm_tete_50_epoch_losses": "orange",
    #"arc_ahrm_tete_25_lr1e-3_wd1e-2_epoch_losses": "green"
}

for file_name in file_names:
    loss_datas = []
    with open(f'./AbstractHRM/saved/json/{file_name}.json', "r") as f:
        loss_datas = np.array(json.load(f))
    plt.plot(loss_datas, color=file_names[file_name])

plt.show()
