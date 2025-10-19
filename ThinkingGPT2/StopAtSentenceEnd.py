import nltk
from nltk.tokenize import sent_tokenize
from transformers import StoppingCriteria, StoppingCriteriaList
import string

class StopAtSentenceEnd(StoppingCriteria):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sentence_amount = -1
        self.max_patience = 5
        self.patience = self.max_patience
        self.suspition = False

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the new tokens
        new_tokens_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if self.sentence_amount == -1:
            sentences = sent_tokenize(new_tokens_text)
            self.sentence_amount = len(sentences) + 1

        if self.patience <= 1:

            self.patience = self.max_patience

            sentences = sent_tokenize(new_tokens_text)

            if len(sentences) > self.sentence_amount:
                if self.suspition == True:
                    #self.reset()
                    self.patience = self.max_patience
                    self.suspition = False
                    self.sentence_amount = -1
                    return True
                else:
                    self.suspition = True
            else:
                self.suspition = False
        

        self.patience -= 1
        
        return False

    def reset(self):
        self.patience = self.max_patience
        self.suspition = False
        self.sentence_amount = -1