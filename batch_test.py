import torch, nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import StoppingCriteria, StoppingCriteriaList
from nltk.tokenize import sent_tokenize


model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.to('cuda')
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

texts = [
    "Hello, how are you?",
    "Once upon a time in a land far away...",
    "The quick brown fox jumps over the lazy dog."
]

encodings = tokenizer(texts, return_tensors="pt", padding=True).to('cuda')
sequences = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

outputs = model(**encodings)
logits = outputs.logits
past = outputs.past_key_values

for i in range(100):
    
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    
    for j in range(len(sequences)):
        sequences[j].append(next_tokens[j].item())

    outputs = model(input_ids=next_tokens.unsqueeze(1), past_key_values=past)
    logits = outputs.logits
    past = outputs.past_key_values

for i in range(len(sequences)):
    print(tokenizer.decode(sequences[i]))