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

nltk.download('punkt')
nltk.download('punkt_tab')


pattern_reader = PatternReader.PatternReader()

dataset = load_dataset("openwebtext", trust_remote_code=True)
dataset.shuffle(seed=42)

input_list = []
sentences_all = []
for i in range(128):
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
GPTNodeValue.should_train = True
pattern_reader.calculate_values()

total_time = time.perf_counter() - start_time
print("total_time: " + str(total_time))

with open('txt/output.txt', 'w', encoding="utf-8") as file:
    file.write(str(pattern_reader))
   

