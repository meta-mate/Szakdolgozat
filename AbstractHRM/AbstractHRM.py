import torch
import torch.nn as nn
import sys
sys.path.append("../Szakdolgozat")
from PatternReader import PatternReader, NodeValue
import gc

class HRMNodeValue(NodeValue):

    def __init__(self, value, is_empty=False):
        super().__init__(value, is_empty)

    def calculate_value(self, lesser_nodes, n):
        raise NotImplementedError("Use step() method instead!")
    
    def create_empty(self):
        return HRMNodeValue(None)

    def copy(self):
        return None
    
    def __str__(self):
        return str(self.value.shape)
    
    def step(pattern_reader, block, x):
        
        last_change = pattern_reader.interpretation(HRMNodeValue(x))
        
        node_list = pattern_reader.node_list

        zeros = torch.zeros(x.shape).to(x.device)

        started = False
        for i in range(len(node_list) - 1, 0, -1):
            node = node_list.nodeat(i).value

            if not started:
                if last_change is node.values.first.value:
                    started = True

                    #calculate its value with a zero greater value

                    value = node.values.first.value
                    value.is_empty = False

                    x_lesser = node.get_lesser_value(0).value
                    x_greater = zeros
                    
                    with torch.no_grad():
                        value.value = block(x_lesser, x_greater)

                else:
                    continue

            if i <= 1:
                break

            lesser_node = node.lesser_nodes.last.value
            x_lesser = lesser_node.get_lesser_value(-2).value
            x_greater = node.values.last.value.value
            
            lesser_value = lesser_node.values.last.value
            
            if i == 2:
                lesser_value.value = block(x_lesser.detach(), x_greater)
            else:    
                with torch.no_grad():
                    lesser_value.value = block(x_lesser, x_greater)
            
        index = min(1, len(node_list) - 1)

        return node_list.nodeat(index).value.values.last.value.value


class AbstractHRM(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.pattern_reader = PatternReader()
        self.block = block

    def reset(self):
        del self.pattern_reader
        self.pattern_reader = PatternReader()
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x):
        
        return HRMNodeValue.step(self.pattern_reader, self.block, x)

        '''
        The stopping logic should take into account:
         - the loss, or the non-ambiguity of the output, and its convergence
         - by how much the next steps computation will grow (n * n / 3)
         - the number of new name types, that indicate unique vantage points
        '''
        y = None
        for i in range(5):
            y = HRMNodeValue.step(self.pattern_reader, self.block, x)
        
        self.reset()
        return y

