from PatternReader import PatternReader, NodeValue
import time
import random
import string
import matplotlib.pyplot as plt


def random_chars():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
class IntNodeValue(NodeValue):
    
    modulator = 2

    def __init__(self, value):
        self.value = value

    def derive_implication(self, lesser_values, n):
        
        #result = self.value + lesser_values.last.value.value
        #result = lesser_values.nodeat(n).value.value + lesser_values.nodeat(n + 1).value.value
        result = lesser_values.nodeat(n + 1).value.value - lesser_values.nodeat(n).value.value
        result += IntNodeValue.modulator
        result %= IntNodeValue.modulator

        #self.value = values.nodeat(n + 1).value
        #self.value = random_chars()
        #result = 0
        self.value = result

    def predict(pattern_reader):
        copied = pattern_reader.copy()
        
        last_change = copied.interpretation(IntNodeValue(0))
        copied.calculate_values()

        return IntNodeValue(IntNodeValue.modulator - last_change.value)

    def create_empty(self):
        return IntNodeValue(0)
    
    def copy(self):
        return IntNodeValue(self.value)

    def __str__(self):
        return str(self.value)
    
pattern_reader = PatternReader()

IntNodeValue.modulator = 12
pattern_reader.interpretation(IntNodeValue(0))
pattern_reader.calculate_values()
for i in range(100):
    print(i)
    prediction = IntNodeValue.predict(pattern_reader)
    next_step = prediction.value + 7
    next_step %= IntNodeValue.modulator
    last_change = pattern_reader.interpretation(IntNodeValue(next_step))
    pattern_reader.calculate_values()

    if last_change.value != 7:
        print("error")

print(pattern_reader.node_list.nodeat(0).value.values)

with open('txt/PRTestOutput.txt', 'w', encoding="utf-8") as file:
    file.write(str(pattern_reader))

pattern_reader = PatternReader()

IntNodeValue.modulator = 2
elements = []
for i in range(9):
    elements.append(round(random.random()))

calculation_lenghts = []
square_results = []
for i in range(len(elements)):
    start_time = time.perf_counter()
    element = elements[i]
    pattern_reader.interpretation(IntNodeValue(element))
    delta_time = time.perf_counter() - start_time
    calculation_lenghts.append(len(pattern_reader.node_list))
    square_results.append(i * i / 3)
    #print("element: " + str(element) + " iteration: " + str(i) + " delta_time: " + str(delta_time))

plt.plot(calculation_lenghts, marker=',', color='blue', label='PatternReader')
plt.plot(square_results, marker=',', color='orange', label='Square')
#plt.show()

pattern_reader.calculate_values()
#print(pattern_reader)

sum = 0
for node in pattern_reader.node_list:
    sum += len(node.get_values())

print(sum)

#with open('txt/PRTestOutput.txt', 'w', encoding="utf-8") as file:
#    file.write(str(pattern_reader))

        
        

