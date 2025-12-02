import torch
import torch.nn as nn
import sys
sys.path.append("../")
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
    
    def delete_unusable(pattern_reader):
        for node in pattern_reader.node_list:
            for i in range(len(node.values)):
                if i == 0 or i == len(node.values) - 1:
                    continue
                node.values.nodeat(i).value.value = None
        gc.collect()
    
    def step(pattern_reader, block, x_lowest, x_greatest):
        
        last_change = pattern_reader.interpretation(HRMNodeValue(x_lowest))
        
        node_list = pattern_reader.node_list

        started = False
        for i in range(len(node_list) - 1, 0, -1):
            node = node_list.nodeat(i).value

            if not started:
                if last_change is node.values.first.value:
                    started = True

                    #calculate its value with a zero greater value

                    value = node.values.first.value
                    value.is_empty = False

                    x_lesser = node.get_lesser_value(-2).value
                    x_greater = x_greatest
                    
                    if i == 1:
                        x_lesser = node.get_lesser_value(-1).value
                        value.value = block(x_lesser, x_greater)
                    else:
                        #x_lesser = x_lesser.detach()
                        with torch.no_grad():
                            value.value = block(x_lesser, x_greater)

                    #value.value = block(x_lesser, x_greater)

                else:
                    continue

            if i <= 1:
                break

            lesser_node = node.lesser_nodes.last.value
            x_lesser = lesser_node.get_lesser_value(-2).value
            x_greater = node.values.last.value.value
            
            lesser_value = lesser_node.values.last.value
            
            if i == 2:
                x_lesser = lesser_node.get_lesser_value(-1).value
                lesser_value.value = block(x_lesser, x_greater)
            else:
                #x_lesser = x_lesser.detach()
                with torch.no_grad():
                    lesser_value.value = block(x_lesser, x_greater)
                
            #lesser_value.value = block(x_lesser, x_greater)

        if pattern_reader.pattern_length > 2 and False:
            HRMNodeValue.delete_unusable(pattern_reader)
        
        index = min(1, len(node_list) - 1)

        return node_list.nodeat(index).value.values.last.value.value


class AbstractHRM(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.pattern_reader = PatternReader()
        self.block = block

    def reset(self):
        self.pattern_reader = PatternReader()
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x_lowest, x_greatest):
        return HRMNodeValue.step(self.pattern_reader, self.block, x_lowest, x_greatest)

