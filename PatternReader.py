from llist import dllist, dllistnode

class NodeValue:
    def derive_implication(self, lesser_values, n):
        raise NotImplementedError("Subclasses should implement this!")

    def create_empty(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this!")


class NameValue(NodeValue):

    def __init__(self, value):
        self.value = value

    def derive_implication(self, lesser_values, n):
        return
    
    def create_empty(self):
        return NameValue(False)
    
    def __str__(self):
        if self.value:
            return "1"
        else:
            return "0"
                
                
class Name():

    def __init__(self, value = None):
        
        self.simple_value = None
        self.pattern_reader = None

        if isinstance(value, int):
            self.simple_value = value
        elif isinstance(value, PatternReader):
            self.pattern_reader = value
        else:
            TypeError("Argument must be the correct type")

    def is_simple(self):
        return self.pattern_reader is None

    def get(self, index):
        if self.is_simple():
            if index != 0:
                raise IndexError(f"Index {index} is out of range for singular name.")
            return self.simple_value
        else:
            return self.pattern_reader.node_list.nodeat(index).value

    def __getitem__(self, key):
        return self.get(key)


    def increment(self):
        if self.is_simple():
            self.simple_value += 1
        else:
            self.pattern_reader.interpretation(NameValue(True))


    def __len__(self):
        if self.is_simple():
            return min(self.simple_value, 1)
        else:
            return len(self.pattern_reader.node_list)
        

    def __str__(self):
        if self.is_simple():
            return str(self.simple_value)
        else:
            return str(self.pattern_reader)


class PatternReader:
    class Node:
        def __init__(self, name, lesser_values=None):
            self.name = name
            self.values = dllist()
            self.first_occurences = dllist()
            
            if lesser_values is not None:
                self.lesser_values = lesser_values
            else:
                self.lesser_values = dllist()

            self.has_bigger = False
            self.lesser_from_bigger = None
            self.bigger = self.calculate_bigger()
            self.incremented = None #self.calculate_incremented()


        def name_equals(self, other_name):
            
            name = self.name

            if name.is_simple() != other_name.is_simple():
                return False

            if name.is_simple() and other_name.is_simple():
                return name.simple_value == other_name.simple_value
            
            if len(name) != len(other_name):
                return False

            for i in range(len(name)):
                for j in range(len(name[i].values)):
                    
                    value1 = name[i].values.nodeat(j).value.value
                    value2 = other_name[i].values.nodeat(j).value.value
                    
                    if value1 != value2:
                        return False

            return True



            return name == other_name

            if len(name) != len(other_name):
                return False

            node1 = name.first
            node2 = other_name.first

            while node1 != name.end:
                if node1 != node2:
                    return False
                node1 = node1.next
                node2 = node2.next

            return True
        

        def copy_name(self, copied_name = None):
            
            name = self.name
            
            if name.is_simple():
                return Name(name.simple_value)

            if copied_name == None:
                copied_name = Name(PatternReader())
                
                for i in range(len(name[0].values)):
                    copied_name.pattern_reader.interpretation(NameValue(True))


            for i in range(len(copied_name)):

                i_adapted = i
                while i_adapted < len(name) and not name[i_adapted].name_equals(copied_name[i].name):
                    i_adapted += 1

                for j in range(len(copied_name[i].values)):
                    value = name[i_adapted].values.nodeat(j).value.value
                    copied_name[i].values.nodeat(j).value.value = value
            
            return copied_name


        def calculate_incremented(self):

            incremented = self.copy_name()

            if incremented.is_simple():
                incremented.simple_value += 1
            else:
                incremented.pattern_reader.interpretation(NameValue(True))
                #incremented[1].values.nodeat(-1).value.value = True

            self.incremented = incremented
            return self.incremented


            r = dllist([node for node in self.name])
            r.insert(1, r.first.next)
            return r

        def calculate_bigger(self):

            name = self.name
            self.has_bigger = False

            if name.is_simple():
                if name.simple_value > 1:

                    bigger = Name(PatternReader())

                    for i in range(2):
                        bigger.increment()

                    bigger[1].values.nodeat(0).value.value = True
                    #bigger[1].values.nodeat(1).value.value = True
                    #bigger[2].values.nodeat(0).value.value = True

                    self.has_bigger = True

                    lesser_from_bigger = Name(name.simple_value - 1)

                    self.bigger = lesser_from_bigger
                    self.lesser_from_bigger = lesser_from_bigger
                    return bigger
                else:
                    return None
            else:

                
                bigger = Name(PatternReader())

                bigger_size = 0
                for i in range(1, len(name)):
                    for j in range(len(name[i].values)):
                        value = name[i].values.nodeat(j).value.value
                        lesser_value = name[i].lesser_values.nodeat(j + 1).value.value
                        if lesser_value:
                            if not value:
                                bigger_size = name[i].first_occurences.nodeat(j).value
                                
                                
                                if i == 1:
                                    template = Name(PatternReader())
                                    for k in range(len(name[0].values) - 1):
                                        template.increment()
                                    self.lesser_from_bigger = self.copy_name(template)
                                else:
                                    self.lesser_from_bigger = self.copy_name()
                                    for k in range(j + 1, len(name[i].lesser_values)):
                                        condition = k + 1 >= len(name[i].lesser_values)
                                        if not condition:
                                            condition = not name[i].lesser_values.nodeat(k + 1).value.value
                                        
                                        if condition:
                                            self.lesser_from_bigger[i].lesser_values.nodeat(k).value.value = False
                                            break
                                    
                                break
                        else:
                            break

                for i in range(bigger_size):
                    bigger.increment()

                bigger = self.copy_name(bigger)

                for i in range(1, len(bigger)):
                    for j in range(len(bigger[i].values)):
                        value = bigger[i].values.nodeat(j).value.value
                        lesser_value = bigger[i].lesser_values.nodeat(j + 1).value.value
                        if lesser_value:
                            if not value:
                                bigger[i].values.nodeat(j).value.value = True
                                self.has_bigger = True
                                break
                    if self.has_bigger:
                        break

                self.bigger = bigger
                return self.bigger

            #########################################################

            r = dllist([1])
            self.lesser_from_bigger = dllist([1])

            name = self.name

            self.has_bigger = False

            if len(name) < 2:
                return r

            last_same_index = 0
            for i in range(1, len(name)):
                if name.nodeat(i).value == name.nodeat(i - 1).value:
                    last_same_index = i
                    if i > 1:
                        self.lesser_from_bigger.append(name.nodeat(i).value)
                    self.has_bigger = True
                elif i == 1:
                    continue
                else:
                    break

            for i in range(last_same_index + 1, len(name)):
                r.append(name.nodeat(i).value)
                self.lesser_from_bigger.append(name.nodeat(i).value)

            if last_same_index != 0:
                r.insert(name.nodeat(last_same_index).value + 1, r.first.next)
                print(str(name) + " has bigger: " + str(r) + " so it has lesser: " + str(self.lesser_from_bigger))

            self.bigger = r
            return r

        def bigger_than(self, other_name):
            
            name = self.name

            if name.is_simple() and other_name.is_simple():
                return name.simple_value > other_name.simple_value
            elif not name.is_simple() and other_name.is_simple():
                return True
            elif name.is_simple() and not other_name.is_simple():
                return False

            longer_name = name
            shorter_name = other_name

            is_shorter = len(name) < len(other_name)

            if is_shorter:
                longer_name = other_name
                shorter_name = name
                
            is_shorter_bigger = 0

            for i in range(len(shorter_name)):
                i_adapted = i
                while i_adapted < len(longer_name) and not shorter_name[i].name_equals(longer_name[i_adapted].name):
                    i_adapted += 1

                if i_adapted == len(longer_name):
                    break
                
                shorter_values = shorter_name[i].values
                longer_values = longer_name[i_adapted].values

                for j in range(len(longer_values)):
                    shorter_value = False
                    longer_value = longer_values.nodeat(j).value.value

                    if j < len(shorter_values):
                        shorter_value = shorter_values.nodeat(j).value.value

                    if shorter_value and not longer_value:
                        is_shorter_bigger = 1
                        break
                    elif not shorter_value and longer_value:
                        is_shorter_bigger = -1
                        break

            if is_shorter:
                return is_shorter_bigger > 0
            else:
                return is_shorter_bigger < 0


            return False


            max = 0
            for i in range(len(other_name)):
                if max < other_name.nodeat(i).value:
                    max = other_name.nodeat(i).value

            found_same = False
            for i in range(len(name)):
                if max < name.nodeat(i).value:
                    return True
                if max == name.nodeat(i).value:
                    found_same = True

            if found_same == False:
                return False
            
            while max >= 1:
                count1 = 0
                count2 = 0

                for i in range(len(other_name)):
                    if max == other_name.nodeat(i).value:
                        count1 += 1

                for i in range(len(name)):
                    if max == name.nodeat(i).value:
                        count2 += 1

                if count1 > count2:
                    return False
                if count1 < count2:
                    return True

                max -= 1

            return False

        def get_values(self):
            return self.values
        
        def append_values(self, value, pattern_length):
            self.values.append(value)
            self.first_occurences.append(pattern_length)

        def get_lesser_values(self):
            return self.lesser_values

        def set_lesser_values(self, lesser_values):
            self.lesser_values = lesser_values

        def get_name(self):
            return self.name
        
        def get_bigger(self):
            return self.bigger
        
        def get_has_bigger(self):
            return self.has_bigger
        
        def get_lesser_from_bigger(self):
            return self.lesser_from_bigger
        
        def get_incremented(self):
            if self.incremented == None:
                self.incremented = self.calculate_incremented()
            return self.incremented

        def __str__(self):
            values_str = ""
            values = self.get_values()
            for i in range(len(values)):
                values_str += str(values.nodeat(i).value)
                if i < len(values) - 1:
                    values_str += " | "
            #values_str = values_str.strip()
            #values_str = values_str.replace("\n", "\n\t\t")

            lesser_values_str = ""
            lesser_values = self.get_lesser_values()
            for i in range(len(lesser_values)):
                lesser_values_str += str(lesser_values.nodeat(i).value)
                if i < len(lesser_values) - 1:
                    lesser_values_str += " | "
            #lesser_values_str = lesser_values_str.strip()
            #lesser_values_str = lesser_values_str.replace("\n", "\n\t\t")

            return f"Node(Name: {self.get_name()} Values: {values_str})"
            
            if self.name_equals(dllist([0])) or self.name_equals(dllist([1])) and False:
                return f"Node(\n\n\tName: {self.get_name()}\n\n\tValues:\n\n\t\t{values_str}\n)"
            else: 
                return f"Node(\n\n\tName: {self.get_name()}\n\n\tLesser values:\n\n\t\t{lesser_values_str}\n\n\tValues:\n\n\t\t{values_str}\n)"

    def __init__(self):
        self.node_list = dllist()
        self.pattern_length = 0

    def reset(self):
        self.node_list.clear()
        self.pattern_length = 0

    def interpretation(self, pattern):
        last_change = None
        self.pattern_length += 1

        def append_node(start_index, lesser_value = None): #the argument should be the lesser value
            
            #instead, just add an empty value to the values, and upon creating a new node, make its lesser values the previous nodes values list

            #implication = None
            values = None

            if start_index < len(self.node_list):
                #implication = new_value.derive_implication(self.node_list.nodeat(start_index).value.get_values())
                #self.node_list.nodeat(start_index).value.set_value(new_value)
                #new_value = implication
                
                self.node_list.nodeat(start_index).value.get_lesser_values().append(lesser_value)
                self.node_list.nodeat(start_index).value.append_values(pattern.create_empty(), self.pattern_length)

                start_index += 1

            temp_index = start_index
            
            name_one = Name(1)

            name_zero = Name(0)

            name_to_look = name_one

            for i in range(start_index, len(self.node_list)):
                if self.node_list.nodeat(i).value.name_equals(name_to_look):
                    temp_index = i + 1
                    #implication = new_value.derive_implication(self.node_list.nodeat(i).value.get_values())
                    #self.node_list.nodeat(i).value.set_value(new_value)
                    #new_value = implication

                    self.node_list.nodeat(i).value.append_values(pattern.create_empty(), self.pattern_length)
                else:
                    break

            if temp_index > 0:
                values = self.node_list.nodeat(temp_index - 1).value.values

            if temp_index < len(self.node_list):
                self.node_list.insert(self.Node(name_one, values), self.node_list.nodeat(temp_index))
                self.node_list.nodeat(temp_index).value.append_values(pattern.create_empty(), self.pattern_length)
            elif len(self.node_list):
                self.node_list.append(self.Node(name_one, values))
                self.node_list.nodeat(temp_index).value.append_values(pattern.create_empty(), self.pattern_length)
            else:
                self.node_list.append(self.Node(name_zero, values))
                self.node_list.nodeat(temp_index).value.append_values(pattern.create_empty(), self.pattern_length)

            return temp_index

        #the 0th node doesnt need lesser values, but its value should be set by the pattern
        temp_index = append_node(0)
        self.node_list.nodeat(0).value.get_values().last.value = pattern

        while True:
            
            is_there_same_below = False
            is_there_bigger_below = False
            
            first_values = dllist()
            name_to_look = self.node_list.nodeat(temp_index).value.get_name()
            
            for i_n in range(temp_index):
                i = temp_index - 1 - i_n

                if self.node_list.nodeat(i).value.name_equals(name_to_look):
                    is_there_same_below = True
                    first_values.appendleft(self.node_list.nodeat(i).value.get_values().first.value)
                    break
                    #enough just once, can break here, save only the previous value
                if self.node_list.nodeat(i).value.bigger_than(name_to_look):
                    #also save if there is bigger below, so we know if this is the first
                    is_there_bigger_below = True
                    break
                

            #if is_there_same_below == True:
                #last_value_implication = self.node_list.nodeat(temp_index).value.get_first_value().derive_implication(first_values)
                #no need to calculate implication

            if is_there_same_below == False:
                if self.node_list.nodeat(temp_index).value.get_has_bigger() == False:
                    last_change = self.node_list.nodeat(temp_index).value.values.first.value
                    break

            
                #name_to_look = self.node_list.nodeat(temp_index).value.get_name()
                lesser_name = self.node_list.nodeat(temp_index).value.get_lesser_from_bigger()
                bigger = self.node_list.nodeat(temp_index).value.get_bigger()

                is_lesser_from_bigger_there = False
                lesser_index = temp_index
                first_values.clear()

                for i_n in range(temp_index):
                    i = temp_index - 1 - i_n

                    if self.node_list.nodeat(i).value.bigger_than(bigger):
                        break
                    if self.node_list.nodeat(i).value.name_equals(bigger):
                        break

                    if self.node_list.nodeat(i).value.bigger_than(name_to_look):
                        is_lesser_from_bigger_there = False
                        break
                    if self.node_list.nodeat(i).value.name_equals(name_to_look):
                        is_lesser_from_bigger_there = False
                        break

                    if self.node_list.nodeat(i).value.name_equals(lesser_name):
                        is_lesser_from_bigger_there = True
                        lesser_index = i

                if is_lesser_from_bigger_there == False:
                    break

                first_values.appendleft(self.node_list.nodeat(lesser_index).value.get_values().first.value)

                name_to_look = self.node_list.nodeat(temp_index).value.get_bigger()
            else:
                name_to_look = self.node_list.nodeat(temp_index).value.get_incremented()

            is_there_already = False

            index_to_put = temp_index + 1
            while index_to_put < len(self.node_list):
                
                if self.node_list.nodeat(index_to_put).value.name_equals(name_to_look):
                    is_there_already = True
                    break
                if self.node_list.nodeat(index_to_put).value.bigger_than(name_to_look):
                    break

                index_to_put += 1

            first_values.append(self.node_list.nodeat(temp_index).value.get_values().first.value)

            if is_there_already == True:
                temp_index = append_node(index_to_put, first_values.last.value)
            else:
                if index_to_put < len(self.node_list):
                    self.node_list.insert(self.Node(name_to_look, first_values), self.node_list.nodeat(index_to_put))
                    self.node_list.nodeat(index_to_put).value.append_values(pattern.create_empty(), self.pattern_length)
                else:
                    self.node_list.append(self.Node(name_to_look, first_values))
                    self.node_list.nodeat(index_to_put).value.append_values(pattern.create_empty(), self.pattern_length)

                temp_index = index_to_put
        
        return last_change


    def calculate_values(self):
        for j in range(1, len(self.node_list)):
            print("level: " + str(j))
            for i, value in enumerate(self.node_list.nodeat(j).value.get_values()):
                lesser_values = self.node_list.nodeat(j).value.get_lesser_values()
                value.derive_implication(lesser_values, i)
                

    def __str__(self):
        return "\n".join(str(node) for node in self.node_list)



