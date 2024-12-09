import PatternReader
import time
import random
import matplotlib.pyplot as plt


class BoolNodeValue:
    
    def __init__(self, value):
        self.value = value

    def derive_implication(self, values):
        
        result = self.value ^ values.last.value.value

        return BoolNodeValue(result)

    def __str__(self):
        return str(self.value)
    
class IntNodeValue:
    
    def __init__(self, value):
        self.value = value

    def derive_implication(self, values):
        
        result = self.value + values.last.value.value
        result %= 2

        return IntNodeValue(result)

    def __str__(self):
        return str(self.value)
    
pattern_reader = PatternReader.PatternReader()

calculation_lenghts = []
square_results = []
for i in range(100):
    start_time = time.perf_counter()
    element = round(random.random())
    pattern_reader.interpretation(IntNodeValue(element))
    delta_time = time.perf_counter() - start_time
    calculation_lenghts.append(len(pattern_reader.node_list))
    square_results.append(i * i / 3)
    print("element: " + str(element) + " iteration: " + str(i) + " delta_time: " + str(delta_time))

plt.plot(calculation_lenghts, marker=',', color='blue', label='PatternReader')
plt.plot(square_results, marker=',', color='orange', label='Square')
plt.show()

print(pattern_reader)

        
        

