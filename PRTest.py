from PatternReader import PatternReader, NodeValue
import time
import random
import string
import matplotlib.pyplot as plt
import gc


def random_chars():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
class IntNodeValue(NodeValue):
    
    modulator = 2

    def __init__(self, value):
        self.value = value

    def calculate_value(self, lesser_values, n):
        
        #result = self.value + lesser_values.last.value.value
        #result = lesser_values.nodeat(n).value.value + lesser_values.nodeat(n + 1).value.value
        
        result = lesser_values.nodeat(n + 1).value.value - lesser_values.nodeat(n).value.value
        result += IntNodeValue.modulator
        result %= IntNodeValue.modulator

        #self.value = lesser_values.nodeat(n + 1).value
        #self.value = random_chars()
        #result = 0
        self.value = result

    def predict(pattern_reader):
        result = None

        if True:    
            copied = pattern_reader.copy()
            
            last_change = copied.interpretation(IntNodeValue(0))
            copied.calculate_values()

            result = IntNodeValue(IntNodeValue.modulator - last_change.value)

        gc.collect()
        return result


    def step_from_top(pattern_reader, step):

        last_change = pattern_reader.interpretation(IntNodeValue(0))
        pattern_reader.calculate_values()
        
        node_list = pattern_reader.node_list

        started = False
        for i in range(len(node_list) - 1, 0, -1):
            node = node_list.nodeat(i).value    
            last_value = node.values.last.value

            if not started:
                if last_change is node.values.first.value:
                    started = True
                    last_value.value = step.value
                else:
                    continue

            new_value = node.get_lesser_value(-2).value + last_value.value
            new_value += IntNodeValue.modulator
            new_value %= IntNodeValue.modulator
            node.get_lesser_value(-1).value = new_value

        return last_change
            

    def create_empty(self):
        return IntNodeValue(0)
    
    def copy(self):
        return IntNodeValue(self.value)

    def __str__(self):
        return str(self.value)


if True:
    pattern_reader = PatternReader()

    IntNodeValue.modulator = 12
    pattern_reader.interpretation(IntNodeValue(0))
    pattern_reader.calculate_values()
    for i in range(13):
        print(i + 2)
        #prediction = IntNodeValue.predict(pattern_reader)
        #next_step = prediction.value + 7

        #if i + 2 == 13:
        #    next_step += 0

        #next_step %= IntNodeValue.modulator
        #last_change = pattern_reader.interpretation(IntNodeValue(next_step))
        #pattern_reader.calculate_values()

        last_change = IntNodeValue.step_from_top(pattern_reader, IntNodeValue(7))

        #if last_change.value != 7:
        #    print("error its ", last_change.value, "at", pattern_reader.pattern_length)

    print(pattern_reader.node_list.nodeat(0).value.values)

    with open('txt/PRTestOutput.txt', 'w', encoding="utf-8") as file:
        file.write(str(pattern_reader))

else:
    pattern_reader = PatternReader()

    IntNodeValue.modulator = 2
    elements = []
    for i in range(13):
        elements.append(round(random.random()))

    calculation_lenghts = []
    square_results = []
    for i in range(len(elements)):
        start_time = time.perf_counter()
        element = elements[i]
        pattern_reader.interpretation(IntNodeValue(i + 1))
        pattern_reader.calculate_values()
        delta_time = time.perf_counter() - start_time
        calculation_lenghts.append(len(pattern_reader.node_list))
        square_results.append(i * i / 3)
        #print("element: " + str(element) + " iteration: " + str(i) + " delta_time: " + str(delta_time))

    plt.plot(calculation_lenghts, marker=',', color='blue', label='PatternReader')
    plt.plot(square_results, marker=',', color='orange', label='Square')
    #plt.show()

    #pattern_reader.calculate_values()
    #print(pattern_reader)

    sum = 0
    for node in pattern_reader.node_list:
        sum += len(node.get_values())

    print(sum)

    with open('txt/PRTestOutput.txt', 'w', encoding="utf-8") as file:
        file.write(str(pattern_reader))

        
        

