import torch
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to('cuda')
model.eval()

# Define the initial input
input_text = "This is a sample sentence with"
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# Number of evaluations to perform
num_evaluations = 150

# Timing inference
start_time = time.time()

with torch.no_grad():
    for _ in range(num_evaluations):
        # Perform a forward pass to get logits
        outputs = model(**inputs)
        
        # Get the most probable next token ID (last token in batch)
        next_token_id = outputs.logits.argmax(-1)[:, -1]
        
        # Decode the token ID to the corresponding text
        next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
        
        # Append the generated token to the input sequence
        input_text += " " + next_token
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# Calculate and print total and per-inference time
end_time = time.time()
total_time = end_time - start_time
avg_time_per_eval = total_time / num_evaluations

print(f"Total time for {num_evaluations} evaluations: {total_time:.2f} seconds")
print(f"Average time per evaluation: {avg_time_per_eval * 1000:.2f} ms")
print(f"Generated text: {input_text}")
