from llist import dllist, dllistnode

class NodeValue:

    def __init__(self, value, is_empty = False):
        self.value = value
        self.is_empty = is_empty

    def calculate_value(self, lesser_nodes, n):
        raise NotImplementedError("Subclasses should implement this!")
    
    def calculate_if_needed(self, lesser_nodes, n):
        if self.is_empty:
            self.is_empty = False
            self.calculate_value(lesser_nodes, n)
            return True
        else:
            return False

    def create_empty(self):
        raise NotImplementedError("Subclasses should implement this!")

    def copy(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this!")


class NameValue(NodeValue):

    def __init__(self, value):
        super().__init__(value)

    def calculate_value(self, lesser_nodes, n):
        return
    
    def create_empty(self):
        return NameValue(False)
    
    def copy(self):
        return NameValue(self.value)
    
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

        self.has_greater = False
        self.lesser_from_greater = None
        self.greater = None
        self.incremented = None


    def is_simple(self):
        return self.pattern_reader is None

    def simplify(self):

        if self.is_simple():
            return self

        if self.pattern_reader.pattern_length == 2:
            if self[1].values.first.value.value:
                #self = Name(1)
                self.simple_value = 1
                self.pattern_reader = None
                return self
        elif self.pattern_reader.pattern_length == 3:
            for i in range(1, len(self)):
                for j in range(len(self[i].values)):
                   if i != 2 and j != 0:
                       if self[i].values.nodeat(j).value.value:
                           return self

            #self = Name(-1)
            self.simple_value = -1
            self.pattern_reader = None
            return self

        return self

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
            if self.simple_value > 0:
                self.simple_value += 1
            else:
                self.pattern_reader = PatternReader()
                for i in range(3):
                    self.pattern_reader.interpretation(NameValue(False))
                
                node_list = self.pattern_reader.node_list
                node_list.nodeat(1).value.values.nodeat(0).value.value = True
                node_list.nodeat(2).value.values.nodeat(0).value.value = True
                        
        else:
            i = 0
            values = self.pattern_reader.node_list.nodeat(1).value.values
            while True:

                while i >= len(values):
                    self.pattern_reader.interpretation(NameValue(False))

                value = values.nodeat(i).value
                if not value.value:
                    value.value = True
                    break

                i += 1


    def get_incremented(self):
        if self.incremented is None:
            self.incremented = self.copy()
            self.incremented.increment()
        return self.incremented
    
    def get_has_greater(self):
        if self.greater is None:
            self.calculate_greater()
        return self.has_greater

    def get_greater(self):
        if self.greater is None:
            self.calculate_greater()
        return self.greater
    
    def get_lesser_from_greater(self):
        if self.lesser_from_greater is None:
            self.calculate_greater()
        return self.lesser_from_greater

    def __eq__(self, other_name):
            
        if self.is_simple() != other_name.is_simple():
            return False

        if self.is_simple() and other_name.is_simple():
            return self.simple_value == other_name.simple_value
        
        if len(self) != len(other_name):
            return False

        for i in range(len(self)):
            for j in range(len(self[i].values)):
                
                value1 = self[i].values.nodeat(j).value.value
                value2 = other_name[i].values.nodeat(j).value.value
                
                if value1 != value2:
                    return False

        return True


    def copy(self, copied_name = None):
        
        if self.is_simple():
            return Name(self.simple_value)
        else:
            copied = None
            if copied_name is not None:
                copied = copied_name.pattern_reader

            return Name(self.pattern_reader.copy(copied))

        if copied_name is None:
            copied_name = Name(PatternReader())
            
            for i in range(len(name[0].values)):
                copied_name.increment()


        for i in range(len(copied_name)):

            i_adapted = i
            while i_adapted < len(name) and not name[i_adapted].name == copied_name[i].name:
                i_adapted += 1

            for j in range(len(copied_name[i].values)):
                value = name[i_adapted].values.nodeat(j).value.value
                copied_name[i].values.nodeat(j).value.value = value
        
        return copied_name


    def calculate_greater(self):
        
        self.has_greater = False

        if self.is_simple():
            if self.simple_value > 1:

                self.has_greater = True
                
                greater = Name(-1)

                lesser_from_greater = Name(self.simple_value - 1)

                self.greater = greater
                self.lesser_from_greater = lesser_from_greater
                return self.greater
            else:
                return None
        else:

            
            #greater = Name(PatternReader())
            greater = self.copy()

            greater_size = 0
            for i in range(2, len(greater)):
                
                #Just look at the first 2 lesser values, if both is True, add a new True to the values, and take 1 from the lessers

                first_lesser = greater[i].get_lesser_value(0).value
                second_lesser = greater[i].get_lesser_value(1).value

                if first_lesser and second_lesser:
                    
                    node = greater[i]
                    lesser_length = node.get_lesser_length()

                    for j in range(1, lesser_length):
                        if not node.get_lesser_value(j).value or j == lesser_length - 1:

                            template = None

                            min_size = 0

                            for k in range(i + 1, len(greater)):
                                for l in range(len(greater[k].values)):
                                    if greater[k].values.nodeat(l).value.value:
                                        min_size = greater[k].first_occurences.nodeat(l).value
                                    else:
                                        break

                            lesser_from_greater_size = node.lesser_nodes.nodeat(j - 1).value.first_occurences.first.value
                            if lesser_from_greater_size > min_size:
                                min_size = lesser_from_greater_size

                            if greater.pattern_reader.pattern_length > min_size:
                                template = Name(PatternReader())
                                for k in range(min_size):
                                    template.pattern_reader.interpretation(NameValue(False))
                        
                            self.lesser_from_greater = self.copy(template)
                            self.lesser_from_greater.simplify()
                            break
                    
                    j = 0
                    while True:
                        
                        while j >= len(node.values):
                            greater.pattern_reader.interpretation(NameValue(False))

                        if not node.values.nodeat(j).value.value:
                            node.values.nodeat(j).value.value = True
                            greater_size = node.first_occurences.nodeat(j).value
                            
                            k = 0
                            while greater[k] != node:
                                
                                for l in range(len(greater[k].values)):
                                    greater[k].values.nodeat(l).value.value = False

                                k += 1

                            if greater_size != greater.pattern_reader.pattern_length:
                                template = Name()
                                for k in range(greater_size):
                                    template.pattern_reader.interpretation(NameValue(False))
                                greater = greater.copy(template)
                                
                            
                            self.has_greater = True
                            self.greater = greater
                            return greater

                        j += 1


    def __gt__(self, other_name):
        
        if self.is_simple() and other_name.is_simple():
            if self.simple_value < 0 and other_name.simple_value >= 0:
                return True
            elif other_name.simple_value < 0:
                return False

            return self.simple_value > other_name.simple_value
        elif not self.is_simple() and other_name.is_simple():
            return True
        elif self.is_simple() and not other_name.is_simple():
            return False

        longer_name = self
        shorter_name = other_name

        is_shorter = len(self) < len(other_name)

        if is_shorter:
            longer_name = other_name
            shorter_name = self
            
        is_shorter_greater = 0

        for i in range(len(shorter_name)):
            i_adapted = i
            while i_adapted < len(longer_name) and not shorter_name[i].name == longer_name[i_adapted].name:
                i_adapted += 1
            
            shorter_values = shorter_name[i].values
            longer_values = longer_name[i_adapted].values

            for j in range(len(longer_values)):
                shorter_value = False
                longer_value = longer_values.nodeat(j).value.value

                if j < len(shorter_values):
                    shorter_value = shorter_values.nodeat(j).value.value

                if shorter_value and not longer_value:
                    is_shorter_greater = 1
                    break
                elif not shorter_value and longer_value:
                    is_shorter_greater = -1
                    break

        if is_shorter:
            return is_shorter_greater > 0
        else:
            return is_shorter_greater < 0

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
        def __init__(self, name, lesser_nodes=None):
            self.name = name
            self.values = dllist()
            self.first_occurences = dllist()
            
            if lesser_nodes is not None:
                self.lesser_nodes = lesser_nodes
            else:
                self.lesser_nodes = dllist()
        
        def append_values(self, value, pattern_length):
            self.values.append(value)
            self.first_occurences.append(pattern_length)

        def get_lesser_value(self, index):

            if self.name == Name(1):
                return self.lesser_nodes.nodeat(0).value.values.nodeat(index).value
            else:
                return self.lesser_nodes.nodeat(index).value.values.first.value

        def get_lesser_length(self):
            return len(self.values) + 1

        def get_name(self):
            return self.name
        

        def __str__(self):
            values_str = ""
            values = self.values
            for i in range(len(values)):
                values_str += str(values.nodeat(i).value)
                if i < len(values) - 1:
                    values_str += " | "
            #values_str = values_str.strip()
            #values_str = values_str.replace("\n", "\n\t\t")

            if self.name != Name(0):
                lesser_values_str = ""
                lesser_length = self.get_lesser_length()
                for i in range(lesser_length):
                    lesser_values_str += str(self.get_lesser_value(i))
                    if i < lesser_length - 1:
                        lesser_values_str += " | "
            #lesser_values_str = lesser_values_str.strip()
            #lesser_values_str = lesser_values_str.replace("\n", "\n\t\t")

            if isinstance(self.values.nodeat(0).value, NameValue) or True:
                return f"Node(Name: {self.name} Values: {values_str})"
            
            if self.name == (Name(0)) or self.name == (Name(1)) and False:
                return f"Node(\n\n\tName: {self.name}\n\n\tValues:\n\n\t\t{values_str}\n)"
            else: 
                return f"Node(\n\n\tName: {self.name}\n\n\tLesser values:\n\n\t\t{lesser_values_str}\n\n\tValues:\n\n\t\t{values_str}\n)"

    def __init__(self):
        self.node_list = dllist()
        self.pattern_length = 0

    def reset(self):
        self.node_list.clear()
        self.pattern_length = 0

    def interpretation(self, pattern):
        last_change = None
        self.pattern_length += 1

        def append_node(start_index, lesser_node = None): 
            
            lesser_nodes = dllist()

            if start_index < len(self.node_list):
                
                self.node_list.nodeat(start_index).value.lesser_nodes.append(lesser_node)
                self.node_list.nodeat(start_index).value.append_values(pattern.create_empty(), self.pattern_length)

                start_index += 1

            temp_index = start_index
            
            name_one = Name(1)

            name_zero = Name(0)

            name_to_look = name_one

            for i in range(start_index, len(self.node_list)):
                if self.node_list.nodeat(i).value.name == name_to_look:
                    temp_index = i + 1

                    self.node_list.nodeat(i).value.append_values(pattern.create_empty(), self.pattern_length)
                else:
                    break

            if temp_index > 0:
                lesser_nodes.append(self.node_list.nodeat(temp_index - 1).value)

            if temp_index < len(self.node_list):
                self.node_list.insert(self.Node(name_one, lesser_nodes), self.node_list.nodeat(temp_index))
            elif len(self.node_list):
                self.node_list.append(self.Node(name_one, lesser_nodes))
            else:
                self.node_list.append(self.Node(name_zero, lesser_nodes))
            
            self.node_list.nodeat(temp_index).value.append_values(pattern.create_empty(), self.pattern_length)

            return temp_index

        temp_index = append_node(0)
        self.node_list.nodeat(0).value.values.last.value = pattern

        while True:
            
            last_change = self.node_list.nodeat(temp_index).value.values.last.value

            is_there_same_below = False
            
            lesser_nodes = dllist()
            name_to_look = self.node_list.nodeat(temp_index).value.name
            
            for i_n in range(temp_index):
                i = temp_index - 1 - i_n

                if self.node_list.nodeat(i).value.name == name_to_look:
                    is_there_same_below = True
                    lesser_nodes.appendleft(self.node_list.nodeat(i).value)
                    break
                    #enough just once, can break here, save only the previous value
                if self.node_list.nodeat(i).value.name > name_to_look:
                    break
                

            if not is_there_same_below:
                if not name_to_look.get_has_greater():
                    break

            
                #name_to_look = self.node_list.nodeat(temp_index).value.name
                lesser = name_to_look.get_lesser_from_greater()
                greater = name_to_look.get_greater()

                is_lesser_from_greater_there = False
                lesser_index = temp_index
                lesser_nodes.clear()

                for i_n in range(temp_index):
                    i = temp_index - 1 - i_n

                    if self.node_list.nodeat(i).value.name == greater:
                        break
                    if self.node_list.nodeat(i).value.name > greater:
                        break

                    if self.node_list.nodeat(i).value.name == name_to_look:
                        is_lesser_from_greater_there = False
                        break
                    if self.node_list.nodeat(i).value.name > name_to_look:
                        is_lesser_from_greater_there = False
                        break

                    if self.node_list.nodeat(i).value.name == lesser:
                        is_lesser_from_greater_there = True
                        lesser_index = i

                if not is_lesser_from_greater_there:
                    break

                lesser_nodes.appendleft(self.node_list.nodeat(lesser_index).value)

                name_to_look = greater
            else:
                name_to_look = name_to_look.get_incremented()

            is_there_already = False

            index_to_put = temp_index + 1
            while index_to_put < len(self.node_list):
                
                if self.node_list.nodeat(index_to_put).value.name == name_to_look:
                    is_there_already = True
                    break
                if self.node_list.nodeat(index_to_put).value.name > name_to_look:
                    break

                index_to_put += 1

            lesser_nodes.append(self.node_list.nodeat(temp_index).value)

            if is_there_already == True:
                temp_index = append_node(index_to_put, lesser_nodes.last.value)
            else:
                if index_to_put < len(self.node_list):
                    self.node_list.insert(self.Node(name_to_look, lesser_nodes), self.node_list.nodeat(index_to_put))
                else:
                    self.node_list.append(self.Node(name_to_look, lesser_nodes))
                self.node_list.nodeat(index_to_put).value.append_values(pattern.create_empty(), self.pattern_length)

                temp_index = index_to_put
        

        return last_change


    def calculate_values(self):
        for i in range(1, len(self.node_list)):
            #print("level: " + str(i))
            lesser_length = self.node_list.nodeat(i).value.get_lesser_length()
            
            lesser_values = dllist()
            for j in range(lesser_length):
                lesser_values.append(self.node_list.nodeat(i).value.get_lesser_value(j))

            values = self.node_list.nodeat(i).value.values
            for j in range(len(values) - 1, -1, -1):
                value = values.nodeat(j).value
                if not value.calculate_if_needed(lesser_values, j):
                    break
    
                
    def copy(self, copied = None):
        
        node_list = self.node_list

        if copied is None:
            copied = PatternReader()
            
            values = node_list.nodeat(0).value.values
            for i in range(len(values)):
                copied.interpretation(values.nodeat(i).value.copy())


        for i in range(len(copied.node_list)):

            i_adapted = i
            while i_adapted < len(node_list) and not node_list.nodeat(i_adapted).value.name == copied.node_list.nodeat(i).value.name:
                i_adapted += 1

            for j in range(len(copied.node_list.nodeat(i).value.values)):
                value = node_list.nodeat(i_adapted).value.values.nodeat(j).value.copy()
                copied.node_list.nodeat(i).value.values.nodeat(j).value = value
        
        return copied

    def __str__(self):
        return "\n".join(str(node) for node in self.node_list)



