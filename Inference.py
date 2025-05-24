import torch, nltk
from accelerate import init_empty_weights, Accelerator
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import StoppingCriteria, StoppingCriteriaList
from nltk.tokenize import sent_tokenize

from StopAtSentenceEnd import StopAtSentenceEnd

import time

accelerator = Accelerator()
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model, tokenizer = accelerator.prepare(model, tokenizer)

device = accelerator.device

model.eval()

#nltk.download('punkt')
#nltk.download('punkt_tab')

# Use the custom stopping criteria
stop_criteria = StopAtSentenceEnd(tokenizer)

with open('txt/InferenceInput.txt', 'r', encoding="utf-8") as file:
    input_read = file.read()

input_text = input_read

start_time = time.perf_counter()

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

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
        temperature=1,                # Lower temperature for more coherent sentences
        eos_token_id=tokenizer.eos_token_id,  # Stops generation at end-of-sequence token
        stopping_criteria=StoppingCriteriaList([stop_criteria])
    )

generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]  # Slice to keep only new tokens
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

total_time = time.perf_counter() - start_time

print("input length: " + str(len(input_text)) + " word count: " + str(len(input_text.split())))
print("total_time: " + str(total_time))
print(generated_text)