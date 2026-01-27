import torch
import torch.nn as nn
import sys
sys.path.append("../")
sys.path.append("../Szakdolgozat")
from PatternReader import PatternReader, NodeValue
import gc

class HRMNodeValue(NodeValue):

    grad_only_lowest = False

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
                if i == 0 or i >= len(node.values) - 2:
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
                    x_previous = x_lowest
                    
                    if i == 1:
                        x_lesser = node.get_lesser_value(-1).value
                        if HRMNodeValue.grad_only_lowest:
                            value.value = block(x_lesser, x_greater, x_previous)
                    else:
                        if HRMNodeValue.grad_only_lowest:
                            with torch.no_grad():
                                value.value = block(x_lesser, x_greater, x_previous)
                        else:
                            x_lesser = x_lesser.detach()

                    if not HRMNodeValue.grad_only_lowest:
                        value.value = block(x_lesser, x_greater, x_previous)

                else:
                    continue

            if i <= 1:
                break

            lesser_node = node.lesser_nodes.last.value
            x_lesser = lesser_node.get_lesser_value(-2).value
            x_greater = node.values.last.value.value
            
            lesser_value = lesser_node.values.last.value

            x_previous = x_lowest
            if lesser_node.get_lesser_length() > 2 and False:
                x_previous = lesser_node.get_lesser_value(-3).value
            
            if i == 2:
                x_lesser = lesser_node.get_lesser_value(-1).value
                #x_previous = None
                if HRMNodeValue.grad_only_lowest:
                    lesser_value.value = block(x_lesser, x_greater, x_previous)
            else:
                if HRMNodeValue.grad_only_lowest:
                    with torch.no_grad():
                        lesser_value.value = block(x_lesser, x_greater, x_previous)
                else:
                    x_lesser = x_lesser.detach()
                
            if not HRMNodeValue.grad_only_lowest:
                lesser_value.value = block(x_lesser, x_greater, x_previous)

        if pattern_reader.pattern_length > 2:
            HRMNodeValue.delete_unusable(pattern_reader)
        
        index = min(1, len(node_list) - 1)

        return node_list.nodeat(index).value.values.last.value.value


class AbstractHRM(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.pattern_reader = PatternReader()
        self.block = block

        self.L_state = None
        self.H_state = None

        self.name = ""
        if self.block.option == 0:
            self.name = "AbstractHRM"
        elif self.block.option == 1:
            self.name = "Transformer"
        elif self.block.option == 2:
            self.name = "TRM"
        elif self.block.option == 3:
            self.name = "TRM_regular"

    def reset(self):
        self.pattern_reader = PatternReader()
        self.L_state = None
        self.H_state = None
        gc.collect()
        torch.cuda.empty_cache()

        print("reasoning_calls", self.block.calls, self.block.time)
        self.block.calls = 0
        self.block.time = 0

    def forward(self, x_lowest, x_greatest):
        if self.block.option == 0:
            return HRMNodeValue.step(self.pattern_reader, self.block, x_lowest, x_greatest)

        self.pattern_reader.pattern_length += 1

        if self.block.option == 1:
            if self.H_state is None:
                self.L_state = torch.zeros_like(x_lowest)
                self.H_state = torch.zeros_like(x_greatest)

            cycles = 21

            self.L_state = self.L_state.detach()
            for i in range(cycles):
                x_lesser = x_lowest
                x_greater = self.L_state
                if i < cycles - 1 and HRMNodeValue.grad_only_lowest:
                    with torch.no_grad():
                        self.L_state = self.block(x_lesser, x_greater)
                else:
                    self.L_state = self.block(x_lesser, x_greater)

            return self.L_state
        
        elif self.block.option == 2:
            if self.H_state is None:
                self.L_state = torch.zeros_like(x_lowest)
                self.H_state = torch.zeros_like(x_greatest)

            L_cycles = 2
            H_cycles = 7

            self.L_state = self.L_state.detach()
            self.H_state = self.H_state.detach()

            for i in range(H_cycles):
                x_lesser = self.H_state
                x_greater = self.L_state
                x_previous = x_greatest
                if i < H_cycles - 1 and HRMNodeValue.grad_only_lowest:
                    with torch.no_grad():
                        self.H_state = self.block(x_lesser, x_greater, x_previous)
                else:
                    self.H_state = self.block(x_lesser, x_greater, x_previous)

                for j in range(L_cycles):
                    x_lesser = self.L_state
                    x_greater = self.H_state
                    x_previous = x_lowest
                    if (j < L_cycles - 1 or i < H_cycles - 1) and HRMNodeValue.grad_only_lowest:
                        with torch.no_grad():
                            self.L_state = self.block(x_lesser, x_greater, x_previous)
                    else:
                        self.L_state = self.block(x_lesser, x_greater, x_previous)
                    
            return self.L_state
        
        elif self.block.option == 3:
            if self.H_state is None:
                self.L_state = torch.zeros_like(x_lowest)
                self.H_state = torch.zeros_like(x_greatest)

            L_cycles = 2
            H_cycles = 7

            self.L_state = self.L_state.detach()
            self.H_state = self.H_state.detach()

            for i in range(H_cycles):
                for j in range(L_cycles):
                    x_lesser = self.L_state
                    x_greater = self.H_state
                    x_previous = x_lowest
                    if (j < L_cycles - 1 or i < H_cycles - 1) and HRMNodeValue.grad_only_lowest:
                        with torch.no_grad():
                            self.L_state = self.block(x_lesser, x_greater, x_previous)
                    else:
                        self.L_state = self.block(x_lesser, x_greater, x_previous)

                x_lesser = self.H_state
                x_greater = self.L_state
                x_previous = x_greatest
                if i < H_cycles - 1 and HRMNodeValue.grad_only_lowest:
                    with torch.no_grad():
                        self.H_state = self.block(x_lesser, x_greater, x_previous)
                else:
                    self.H_state = self.block(x_lesser, x_greater, x_previous)
                    
            return self.H_state
