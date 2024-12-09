import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import StoppingCriteria, StoppingCriteriaList

import PatternReader
import string
import time
import random


option = 3
# Load tokenizer and model
if option == 0:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
elif option == 1:
    tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = BertForMaskedLM.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
elif option == 2:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
elif option == 3:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')    

model.to('cuda')
model.eval()


class StopAtSentenceEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last token generated
        last_token = self.tokenizer.decode(input_ids[0][-1])
        # Stop if the last token ends with a sentence terminator
        if last_token in [".", "!", "?"]:
            return True
        return False

# Use the custom stopping criteria
stop_criteria = StopAtSentenceEnd(tokenizer)

class BertNodeValue:
    
    def __init__(self, value):
        self.value = value

    def inference(inputText):
        encoded_input = tokenizer(inputText, return_tensors='pt').to('cuda')

        # Forward pass to get model predictions
        with torch.no_grad():
            output = model(**encoded_input)

        # Locate the index of the [MASK] token
        mask_token_index = torch.where(encoded_input['input_ids'] == tokenizer.mask_token_id)[1]

        # Get the logits for the [MASK] token and find the top predicted token
        mask_token_logits = output.logits[0, mask_token_index, :]
        top_token_id = mask_token_logits.argmax(dim=-1).item()

        # Decode the predicted token
        predicted_token = tokenizer.decode([top_token_id])

        return predicted_token


    def derive_implication(self, values):
        
        input_sentence = ""
        result = ""
        
        for value in values:
            input_sentence += value.value + " "

        input_sentence += self.value
        last_added_string = self.value
        
        for i in range(5):
            input_sentence_with_mask = input_sentence + " [MASK]"
            
            if last_added_string[-1] not in string.punctuation:
                input_sentence_with_mask += '.'

            last_added_string = BertNodeValue.inference(input_sentence_with_mask)
            input_sentence += " " + last_added_string
            result += last_added_string + " "

        if result[-1] in string.whitespace:
            result = result[:-1]

        return BertNodeValue(result)

    def __str__(self):
        return self.value
    
class GPTNodeValue:
    
    def __init__(self, value):
        self.value = value

    def inference(input_text, amount_to_generate = 5):

        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')

        # Forward pass to get model predictions
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,              # Large enough to complete sentences
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,                # Lower temperature for more coherent sentences
                eos_token_id=tokenizer.eos_token_id,  # Stops generation at end-of-sequence token
                stopping_criteria=StoppingCriteriaList([stop_criteria])
            )

        generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]  # Slice to keep only new tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                                  
        return generated_text

    def derive_implication(self, values):
        
        input_sentence = ""
        result = ""
        
        for value in values:
            input_sentence += value.value
            if len(input_sentence) > 0:
                if input_sentence[-1] not in string.whitespace:
                    input_sentence += " "

        input_sentence += self.value
        last_added_string = self.value
        
        last_added_string = GPTNodeValue.inference(input_sentence)
        #input_sentence += " " + last_added_string
        result = last_added_string

        print("amount generated: " + str(len(result.split())))
        return GPTNodeValue(result)

    def __str__(self):
        return self.value
    
pattern_reader = PatternReader.PatternReader()

#input_list = ['it', 'was', 'a', 'good', 'movie']
#input_list = ['it', 'was', 'a', 'good', 'movie', 'i', 'have', 'to', 'admit', 'i', 'liked', 'it', 'very', 'much']

with open('txt/input.txt', 'r', encoding="utf-8") as file:
    input_read = file.read()

def whitespace_around_punctuation(input_read, before_too = False):
    input_str = ""
    for i in range(len(input_read)):
        if i > 0:
            if input_read[i] in string.punctuation:   
                if input_read[i - 1] not in string.whitespace:
                    if before_too:
                        input_str += " "

        input_str += input_read[i]

        if i < len(input_read) - 1:
            if input_read[i] in string.punctuation:   
                if input_read[i + 1] not in string.whitespace:
                    input_str += " "

    return input_str

input_str = whitespace_around_punctuation(input_read)

#input_list = input_str.split()

input_list = []

def split_to_sentences(input_str):
    i_start = 0
    for i in range(len(input_str)):
        if input_str[i] in [".", "!", "?"]:
            input_list.append(input_str[i_start : i + 1])
            i_start = i + 1

    lengths = []
    sum = 0
    max = 0
    for i in range(len(input_list)):
        length = len(input_list[i].split())
        lengths.append(length)
        sum += length
        if max < length:
            max = length

    print(lengths)
    print('average: ' + str(sum / len(lengths)))
    print('max: ' + str(max))

split_to_sentences(input_str)

print(len(input_list))

i = 0
string_to_process = ""
for element in input_list:
    string_to_process = element + ''
    if i % 1 == 0 and i >= 0:
        #string_to_process = string_to_process[:-1]
        start_time = time.perf_counter()
        if option == 3:
            pattern_reader.interpretation(GPTNodeValue(string_to_process))
        else:
            pattern_reader.interpretation(BertNodeValue(string_to_process))
        delta_time = time.perf_counter() - start_time
        print(string_to_process + " i: " + str(i) + " delta time: " + str(delta_time))
        string_to_process = ""
    i += 1

#print(pattern_reader)

with open('txt/output.txt', 'w', encoding="utf-8") as file:
    file.write(str(pattern_reader))

        
        

