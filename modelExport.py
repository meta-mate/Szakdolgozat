import torch
from transformers import BertModel


model_name = "bert-base-uncased"
model_path = "./" + model_name + ".onnx"
#model = BertBaseUncased.from_pretrained(model_name)

model = BertModel.from_pretrained(model_name)

# set the model to inference mode
# It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, 
# to turn the model to inference mode. This is required since operators like dropout or batchnorm 
# behave differently in inference and training mode.
model.eval()

# Generate dummy inputs to the model. Adjust if necessary.
inputs = {
        # list of numerical ids for the tokenized text
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long), 
        # dummy list of ones
        'attention_mask': torch.ones([1, 32], dtype=torch.long),     
        # dummy list of ones
        'token_type_ids':  torch.ones([1, 32], dtype=torch.long)     
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model,                                         
# model being run
                  (inputs['input_ids'],
                   inputs['attention_mask'], 
                   inputs['token_type_ids']),                    # model input (or a tuple for multiple inputs)
                  model_path,                                    # where to save the model (can be a file or file-like object)
                  opset_version=11,                              # the ONNX version to export the model to
                  do_constant_folding=True,                      # whether to execute constant folding for optimization
                  input_names=['input_ids', 'attention_mask', 'token_type_ids'],                   # the model's input names
                  output_names=['last_hidden_state', 'pooler_output'],   # the model's output names
                  dynamic_axes={'input_ids': symbolic_names,
                                'input_mask' : symbolic_names,
                                'segment_ids' : symbolic_names,
                                'start_logits' : symbolic_names, 
                                'end_logits': symbolic_names})   # variable length axes/dynamic input
