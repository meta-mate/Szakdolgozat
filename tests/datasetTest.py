from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import string

# Load WikiText-2 (smaller version) or WikiText-103 (larger version)
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")

# Check the splits: 'train', 'validation', 'test'
#print(wikitext)

# Load the dataset
#dataset = load_dataset("openwebtext", trust_remote_code=True)

# Access a few training samples
#for i in range(5):
    #print(f"Sample {i + 1}:\n{dataset['train'][i]['text']}\n")

# Access the training data
train_texts = wikitext['train']['text']

def clean(text):
    text = text.replace(" ", "", 1)
    text = text.replace(" \\'", "'")
    text = text.replace(" '", "'")
    text = text.replace(" ' ", "'")
    apostrophe_dict = {"\"" : True, "`" : True}
    stop_punct = [".", "!", "?", ",", ";", ":"]
    opening_punct = ["(", "[", "{", "<"]
    closing_punct = [")", "]", "}", ">"]
    slash_punct = ["\\", "/"]
    exception_punct = ["-"]
    
    i = 0
    while i < (len(text)):
        if text[i] in apostrophe_dict.keys():
            if apostrophe_dict[text[i]]:
                if text[i + 1] == " ":
                    text = text[: i + 1] + text[i + 2:]
            else:
                if text[i - 1] == " ":
                    text = text[: i - 1] + text[i:]
                    i -= 1
            apostrophe_dict[text[i]] = not apostrophe_dict[text[i]]
        i += 1

    for punct in string.punctuation:
        text = text.replace(f" @{punct}@ ", f"{punct}")
        
        if punct in exception_punct:
            continue
        
        if punct in opening_punct:
            text = text.replace(f" {punct} ", f" {punct}")
        elif punct in slash_punct:
            text = text.replace(f" {punct} ", f"{punct}")

        text = text.replace(f" {punct} ", f"{punct} ")   

    
    for whitespace in string.whitespace:
        text = text.replace(f" {whitespace} ", f"{whitespace}")     
        

    return text    

cleaned_texts = []
j = -1
i = -1
while i < len(train_texts) - 1:
    i += 1
    if train_texts[i][:2] == " =":
        
        if j >= 0:
            cleaned_texts[j] = clean(cleaned_texts[j])
        j += 1
        cleaned_texts.append("")
        continue
    
    if j >= 0:
        cleaned_texts[j] += train_texts[i]

# Print a sample
#print(train_texts[:10])  # First 5 samples
#print(cleaned_texts)

print(len(cleaned_texts))
concatenated_text = ""
sentence_lengths = []
above_5_sentences = 0
above_13_sentences = 0
for cleaned_text in cleaned_texts:
    sentences = sent_tokenize(cleaned_text)
    sentence_lengths.append(len(sentences))
    if len(sentences) > 5:
        above_5_sentences += 1
    if len(sentences) > 13:
        above_13_sentences += 1
        concatenated_text += cleaned_text + "\n\n\n"

print("above_5_sentences: " + str(above_5_sentences))
print("above_13_sentences: " + str(above_13_sentences))

#concatenated_text = ""
#for cleaned_text in cleaned_texts:
#    concatenated_text += cleaned_text + "\n\n\n"

with open('txt/dataset_output.txt', 'w', encoding="utf-8") as file:
    file.write(concatenated_text)