import torch, nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import StoppingCriteria, StoppingCriteriaList
from nltk.tokenize import sent_tokenize

import PatternReader
from PatternReader import NodeValue

from StopAtSentenceEnd import StopAtSentenceEnd

import string
import time
import random


model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.to('cuda')
model.eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#nltk.download('punkt')
#nltk.download('punkt_tab')

whitespace = string.whitespace + chr(160)

# Use the custom stopping criteria
stop_criteria = StopAtSentenceEnd(tokenizer)

max_batch_size = 2
    
class GPTNodeValue(NodeValue):
    
    def __init__(self, value = []):
        self.value = value

    def is_acceptable(result, i):

        sentence_end_punctuation = ".!?"

        amount_generated = len(result.split())
        generation_log = "amount generated: " + str(amount_generated)
        if i > 0:
            generation_log += " try: " + str(i + 1)

        repeated_punctuations = ""
        last_punctuation = ""
        repeat_sum = 0
        max_repeat_sum = 0
        for char in result:
            if char in string.punctuation:
                if last_punctuation == char:
                    repeat_sum += 1
                    if len(repeated_punctuations) <= 0 or not repeated_punctuations[-1] == last_punctuation:
                        repeated_punctuations += last_punctuation
                    repeated_punctuations += char
                else:
                    if max_repeat_sum < repeat_sum:
                        max_repeat_sum = repeat_sum
                    repeat_sum = 1
                    last_punctuation = char

        begin_punctuations = ""
        for char in result:
            if char in string.punctuation:
                begin_punctuations += char
            else:
                break

        illegal_chars = ""
        legal_chars = string.ascii_letters + string.digits + whitespace + string.punctuation
        legal_chars += chr(160)
        for char in result:
            if char not in legal_chars:
                illegal_chars += char
        
        should_regenerate = False

        if max_repeat_sum > 3:
            generation_log += " too much repeated punctuations: " + repeated_punctuations
            should_regenerate = True

        if len(begin_punctuations) > 0:
            generation_log += " punctuations at the beginning: " + begin_punctuations
            should_regenerate = True

        if len(illegal_chars) > 0:
            orders = ""
            for char in illegal_chars:
                orders += str(ord(char)) + ", "
            generation_log += " illegal chars: " + illegal_chars + " orders: " + orders
            should_regenerate = True

        if last_punctuation not in sentence_end_punctuation:
            generation_log += " bad ending: " + result[-1]
            should_regenerate = True

        if amount_generated < 2:
            should_regenerate = True

        print(generation_log)

        return not should_regenerate
        

    def inference(input_texts):

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')

        # Forward pass to get model predictions
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,              # Large enough to complete sentences
                #max_length=inputs.input_ids.shape[-1] + 100,  # Current input length + max_new_tokens
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,                # Lower temperature for more coherent sentences
                eos_token_id=tokenizer.eos_token_id,  # Stops generation at end-of-sequence token
                stopping_criteria=StoppingCriteriaList([stop_criteria])
            )

        generated_texts = []
        for output in outputs:
            #generated_tokens = output[inputs.input_ids.shape[-1]:]  # Slice to keep only new tokens
            
            start_index = inputs.input_ids.shape[-1]
            generated_tokens = output[start_index:] if output.shape[0] > start_index else output

            
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
                                  
        return generated_texts


    def derive_implication(self, values, n): #values is 2 dimensional

        input_sequences = {}
        result = []
        batch_size = len(values.nodeat(0).value.value)
        
        for j in range(batch_size):
            input_sequences[j] = ""
            result.append("")
            for i in range(n + 1):
                value = values.nodeat(i).value.value[j]
                if len(input_sequences[j]) > 0:
                    if input_sequences[j][-1] not in whitespace:
                        if len(value) > 0:
                            if value[0] not in whitespace:
                                input_sequences[j] += " "
                input_sequences[j] += value
        
        
        for j in range(len(input_sequences)):
            if len(input_sequences[j]) > 0:
                if input_sequences[j][-1] not in whitespace:
                    input_sequences[j] += " "


        i = 0
        while len(input_sequences) != 0:
            
            inputs_for_inference = []
            keyStr = "keys: "
            for key in input_sequences:
                inputs_for_inference.append(input_sequences[key])
                keyStr += str(key) + " "
            print(keyStr)
            
            
            new_values = GPTNodeValue.inference(inputs_for_inference)
            to_pop = []
            for j, key in enumerate(input_sequences):

                new_value = new_values[j]
                sentences = sent_tokenize(new_value)
                if len(sentences) > 0:
                    new_value = sentences[0]

                if new_value[-1] == chr(160):
                    new_value[-1] = " "
                new_value = new_value.strip()

                if GPTNodeValue.is_acceptable(new_value, i):
                    to_pop.append(key)
                    result[key] = new_value

            for key in to_pop:
                input_sequences.pop(key)
            
            i += 1

        self.value = result

    def create_empty(self):
        return GPTNodeValue([])

    def __str__(self):
        return str(self.value)