import torch, nltk
from nltk.tokenize import sent_tokenize

import PatternReader
from PatternReader import NodeValue

from GPTNodeValue import GPTNodeValue

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
    for i in range(3):
        batch.append(sentence)

    input_list.append(batch)


#print(input_list)
print(len(input_list))

for i in range(len(input_list)):
    if i > 5:
        break
    
    string_to_process = input_list[i]
    pattern_reader.interpretation(GPTNodeValue(string_to_process))
    
start_time = time.perf_counter()

pattern_reader.calculate_values()

total_time = time.perf_counter() - start_time
print("total_time: " + str(total_time))

with open('txt/output.txt', 'w', encoding="utf-8") as file:
    file.write(str(pattern_reader))

        
        

