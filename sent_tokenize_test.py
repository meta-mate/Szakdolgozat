import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

text = "The vector space model (VTM) provides the most widely used solution for the concise representation of textual content. The model describes each document with a vector, each element represents the occurrence of individual terms (usually words). Terms are the units of representation, usually words delimited by punctuation marks (unigrams). Some methods also use multi-word expressions (n-grams) rep. e., which generally characterizes documents better than but increases the time and storage requirements of doc. processing. The vector space model (VTM) provides the most widely used solution for the concise representation of textual content. The model describes each document with a vector, each element represents the occurrence of individual terms (usually words). Terms are the units of representation, usually words delimited by punctuation marks (unigrams). Some methods also use multi-word expressions (n-grams) rep. e., which generally characterizes documents better than but increases the time and storage requirements of document processing.\nWith the development of recent years, deep learning models such as BERT (Bidirectional Encoder Representations from Transformers) and other transformer-based language models offer even more advanced text representation, taking into account context and the relationships between words."
#with open('txt/dataset_output.txt', 'r', encoding="utf-8") as file:
    #text = file.read()
sentences = sent_tokenize(text)
print(len(sentences))
print(sentences)