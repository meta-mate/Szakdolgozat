import torch, nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

import PatternReader
from PatternReader import NodeValue

from GPTNodeValue import GPTNodeValue

import json
import matplotlib.pyplot as plt

import string
import time
import random

#nltk.download('punkt')
#nltk.download('punkt_tab')


pattern_reader = PatternReader.PatternReader()

with open('txt/input.txt', 'r', encoding="utf-8") as file:
    input_read = file.read()

input_str = input_read

sentences = sent_tokenize(input_str)
input_list = []
for sentence in sentences:
    batch = []
    for i in range(10):
        batch.append(sentence)

    input_list.append(batch)


dataset = load_dataset("openwebtext", trust_remote_code=True)

input_list = []
sentences_all = []
for i in range(400 * 5):
    sentences = sent_tokenize(dataset["train"][i]["text"])

    if len(sentences) >= 5:
        sentences_all.append(sentences)

for i in range(5):

    sentences_at_i = []

    for j in range(len(sentences_all)):
        sentences_at_i.append(sentences_all[j][i])

    input_list.append(sentences_at_i)

#print(input_list)
print(len(input_list))

for i in range(len(input_list)):
    if i > 5:
        break
    
    pattern_reader.interpretation(GPTNodeValue(input_list[i]))
    
start_time = time.perf_counter()

pattern_reader.calculate_values()

total_time = time.perf_counter() - start_time
print("total_time: " + str(total_time))

with open('txt/output.txt', 'w', encoding="utf-8") as file:
    file.write(str(pattern_reader))
   

for loss_tracker in GPTNodeValue.loss_trackers:    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_tracker.steps, loss_tracker.losses, marker=',')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)        
        

