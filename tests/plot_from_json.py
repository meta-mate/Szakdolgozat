import json
import matplotlib.pyplot as plt


with open('./AbstractHRM/saved/epoch_losses.json', "r") as f:
    loss_datas = json.load(f)

plt.plot(loss_datas)
plt.show()

'''
for i, loss_data in enumerate(loss_datas):    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_data['steps'], loss_data['losses'], marker=',')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("img/level" + str(i + 1) + ".png")
'''