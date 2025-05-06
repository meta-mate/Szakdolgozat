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

#nltk.download('punkt')
#nltk.download('punkt_tab')

whitespace = string.whitespace + chr(160)

# Use the custom stopping criteria
stop_criteria = StopAtSentenceEnd(tokenizer)
    
class GPTNodeValue(NodeValue):
    
    def __init__(self, value = ""):
        self.value = value

    def inference(input_text):
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')

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

        generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]  # Slice to keep only new tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                                  
        return generated_text

    def derive_implication(self, values, n):

        input_sentences = ""
        result = ""
        
        for i in range(n + 1):
            value = values.nodeat(i).value.value
            if len(input_sentences) > 0:
                if input_sentences[-1] not in whitespace:
                    if len(value) > 0:
                        if value[0] not in whitespace:
                            input_sentences += " "
            input_sentences += value
        
        if len(input_sentences) > 0:
            if input_sentences[-1] not in whitespace:
                input_sentences += " "

        should_regenerate = True
        i = 0
        while should_regenerate:
            new_value = GPTNodeValue.inference(input_sentences)
            sentences = sent_tokenize(new_value)
            if len(sentences) > 0:
                result = sentences[0]

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

            if max_repeat_sum > 2:
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

            if amount_generated < 2:
                should_regenerate = True

            print(generation_log)
            
            i += 1

        result = result.strip()
        if result[-1] in whitespace:
            result = result[:-1]
        #print(input_sentences + " " + result)
        self.value = result

    def create_empty(self):
        return GPTNodeValue("")

    def __str__(self):
        return self.value