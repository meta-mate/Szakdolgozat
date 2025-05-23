import torch, nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from nltk.tokenize import sent_tokenize

import matplotlib.pyplot as plt

import PatternReader
from PatternReader import NodeValue

from StopAtSentenceEnd import StopAtSentenceEnd
from LossTrackerCallback import LossTrackerCallback

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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#nltk.download('punkt')
#nltk.download('punkt_tab')

whitespace = string.whitespace + chr(160)

# Use the custom stopping criteria
stop_criteria = StopAtSentenceEnd(tokenizer)

batch_size = 32
train_batch_size = 8
    
class GPTNodeValue(NodeValue):

    loss_trackers = []
    
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
            generation_log += " bad ending: " + last_punctuation
            should_regenerate = True

        if amount_generated < 2:
            should_regenerate = True

        print(generation_log)

        return not should_regenerate
        
    def train(input_texts):

        loss_tracker = LossTrackerCallback()

        dataset = Dataset.from_dict({"text": input_texts})

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

        tokenized_dataset = dataset.map(tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned",
            overwrite_output_dir=True,
            per_device_train_batch_size=train_batch_size,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            eval_strategy="no",
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[loss_tracker]
        )

        trainer.train()

        GPTNodeValue.loss_trackers.append(loss_tracker)

        json_data = []
        for loss_tracker in GPTNodeValue.loss_trackers:    
            json_data.append({
                "steps": loss_tracker.steps,
                "losses": loss_tracker.losses
            })

        with open('json/losses.json', 'w') as file:
            json.dump(json_data, file) 



    def inference(input_texts):

        batches = []

        i = 0
        while True:

            startIndex = i * batch_size
            endIndex = min((i + 1) * batch_size, len(input_texts))

            if startIndex >= endIndex:
                break

            batches.append(input_texts[startIndex : endIndex])
            
            if startIndex > len(input_texts):
                break

            i += 1


        generated_texts = []
        for batch in batches:

            #print("batch remaining: " + str(i))
            i -= 1

            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')

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

            for output in outputs:
                #generated_tokens = output[inputs.input_ids.shape[-1]:]  # Slice to keep only new tokens
                
                start_index = inputs.input_ids.shape[-1]
                generated_tokens = output[start_index:] if output.shape[0] > start_index else output

                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
                                  
        return generated_texts

    def construct_input_dict(values, n):
        input_dict = {}
        
        for j in range(len(values.nodeat(0).value.value)):
            input_dict[j] = ""
            for i in range(n + 1):
                value = values.nodeat(i).value.value[j]
                if len(input_dict[j]) > 0:
                    if input_dict[j][-1] not in whitespace:
                        if len(value) > 0:
                            if value[0] not in whitespace:
                                input_dict[j] += " "
                input_dict[j] += value
        
        
        for j in range(len(input_dict)):
            if len(input_dict[j]) > 0:
                if input_dict[j][-1] not in whitespace:
                    input_dict[j] += " "

        return input_dict

    def array_from_dict(input_dict):
        result = []
        for key in input_dict:
            result.append(input_dict[key])
        return result


    def derive_implication(self, values, n): #values is 2 dimensional

        input_dict = GPTNodeValue.construct_input_dict(values, n)
        
        result = []
        for j in range(len(input_dict)):
            result.append("")

        if n == 0:
            train_dict = GPTNodeValue.construct_input_dict(values, len(values) - 1)
            train_array = GPTNodeValue.array_from_dict(train_dict)
            GPTNodeValue.train(train_array)


        i = 0
        while len(input_dict) != 0:
            
            keyStr = "keys: "
            for key in input_dict:
                keyStr += str(key) + " "
            print(keyStr)
            
            inputs_for_inference = GPTNodeValue.array_from_dict(input_dict)
            if i > 10:
                for j in range(len(inputs_for_inference)):
                    inputs_for_inference[j] += tokenizer.eos_token

            
            new_values = GPTNodeValue.inference(inputs_for_inference)
            to_pop = []
            for j, key in enumerate(input_dict):

                new_value = new_values[j]
                sentences = sent_tokenize(new_value)
                if len(sentences) > 0:
                    new_value = sentences[0]

                if len(new_value) > 0:
                    if new_value[-1] == chr(160):
                        new_value[-1] = " "
                new_value = new_value.strip()

                if GPTNodeValue.is_acceptable(new_value, i):
                    to_pop.append(key)
                    result[key] = new_value

            for key in to_pop:
                input_dict.pop(key)
            
            i += 1

        self.value = result

    def create_empty(self):
        return GPTNodeValue([])

    def __str__(self):
        return str(self.value[0])