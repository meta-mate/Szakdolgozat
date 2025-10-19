from datasets import load_dataset
from nltk.tokenize import sent_tokenize


# Load OpenWebText with permission to execute custom loading code
dataset = load_dataset("openwebtext", trust_remote_code=True)

#print(len(dataset['train']))

# Print a few samples
for i in range(50):
    print(f"Sample {i + 1}:\n{dataset['train'][i]['text']}\n")
    #print(len(sent_tokenize(dataset['train'][i]['text'])))