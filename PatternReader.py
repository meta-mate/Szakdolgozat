from llist import dllist, dllistnode

class NodeValue:
    def derive_implication(self, lesser_nodes, n):
        raise NotImplementedError("Subclasses should implement this!")

    def create_empty(self):
        raise NotImplementedError("Subclasses should implement this!")

    def copy(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this!")


class NameValue(NodeValue):

    def __init__(self, value):
        self.value = value

    def derive_implication(self, lesser_nodes, n):
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

        self.has_bigger = False
        self.lesser_from_bigger = None
        self.bigger = None
        self.incremented = None


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

    def get_incremented(self):
        if self.incremented is None:
            self.incremented = self.copy()
            self.incremented.increment()
        return self.incremented
    
    def get_has_bigger(self):
        if self.bigger is None:
            self.calculate_bigger()
        return self.has_bigger

    def get_bigger(self):
        if self.bigger is None:
            self.calculate_bigger()
        return self.bigger
    
    def get_lesser_from_bigger(self):
        if self.lesser_from_bigger is None:
            self.calculate_bigger()
        return self.lesser_from_bigger

    def __eq__(self, other_name):
            
        name = self

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


    def copy(self, copied_name = None):
        
        name = self

        if name.is_simple():
            return Name(name.simple_value)
        else:
            copied = None
            if copied_name is not None:
                copied = copied_name.pattern_reader

            result = Name(self.pattern_reader.copy(copied))

            return result

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


    def calculate_bigger(self):

        name = self
        
        name.has_bigger = False

        if name.is_simple():
            if name.simple_value > 1:

                bigger = Name(PatternReader())

                for i in range(2):
                    bigger.increment()

                bigger[1].values.nodeat(0).value.value = True

                name.has_bigger = True

                lesser_from_bigger = Name(name.simple_value - 1)

                name.bigger = bigger
                name.lesser_from_bigger = lesser_from_bigger
                return name.bigger
            else:
                return None
        else:

            
            bigger = Name(PatternReader())

            bigger_size = 0
            for i in range(1, len(name)):
                for j in range(len(name[i].values)):
                    value = name[i].values.nodeat(j).value.value
                    lesser_value = name[i].get_lesser_value(j + 1)
                    lesser_value = lesser_value.value
                    if lesser_value:
                        if not value:
                            bigger_size = name[i].first_occurences.nodeat(j).value
                            
                            if i == 1:
                                template = Name(PatternReader())
                                for k in range(len(name[0].values) - 1):
                                    template.increment()
                                name.lesser_from_bigger = name.copy(template)
                            else:
                                name.lesser_from_bigger = name.copy()
                                lesser_length = name[i].get_lesser_length()
                                for k in range(j + 1, lesser_length):
                                    condition = k + 1 >= lesser_length
                                    if not condition:
                                        condition = not name[i].get_lesser_value(k + 1).value
                                    
                                    if condition:
                                        name.lesser_from_bigger[i].get_lesser_value(k).value = False
                                        break
                                
                            break
                    else:
                        break

            for i in range(bigger_size):
                bigger.increment()

            bigger = name.copy(bigger)

            for i in range(1, len(bigger)):
                for j in range(len(bigger[i].values)):
                    value = bigger[i].values.nodeat(j).value.value
                    lesser_value = bigger[i].get_lesser_value(j + 1).value
                    if lesser_value:
                        if not value:
                            bigger[i].values.nodeat(j).value.value = True
                            name.has_bigger = True
                            break
                if name.has_bigger:
                    break

            name.bigger = bigger
            return name.bigger


    def __gt__(self, other_name):
        
        name = self

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
                    is_shorter_bigger = 1
                    break
                elif not shorter_value and longer_value:
                    is_shorter_bigger = -1
                    break

        if is_shorter:
            return is_shorter_bigger > 0
        else:
            return is_shorter_bigger < 0

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

        def get_values(self):
            return self.values
        
        def append_values(self, value, pattern_length):
            self.values.append(value)
            self.first_occurences.append(pattern_length)

        def get_lesser_value(self, index):

            if self.name == Name(1):
                return self.lesser_nodes.nodeat(0).value.get_values().nodeat(index).value
            else:
                return self.lesser_nodes.nodeat(index).value.get_values().first.value

        def get_lesser_length(self):
            return len(self.values) + 1

        def get_lesser_nodes(self):
            return self.lesser_nodes

        def set_lesser_nodes(self, lesser_nodes):
            self.lesser_nodes = lesser_nodes

        def get_name(self):
            return self.name
        

        def __str__(self):
            values_str = ""
            values = self.get_values()
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

            return f"Node(Name: {self.get_name()} Values: {values_str})"
            
            if self.get_name() == (Name(0)) or self.get_name() == (Name(1)) and False:
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

        def append_node(start_index, lesser_node = None): 
            
            lesser_nodes = dllist()

            if start_index < len(self.node_list):
                
                self.node_list.nodeat(start_index).value.get_lesser_nodes().append(lesser_node)
                self.node_list.nodeat(start_index).value.append_values(pattern.create_empty(), self.pattern_length)

                start_index += 1

            temp_index = start_index
            
            name_one = Name(1)

            name_zero = Name(0)

            name_to_look = name_one

            for i in range(start_index, len(self.node_list)):
                if self.node_list.nodeat(i).value.get_name() == name_to_look:
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
        self.node_list.nodeat(0).value.get_values().last.value = pattern

        while True:
            
            is_there_same_below = False
            
            lesser_nodes = dllist()
            name_to_look = self.node_list.nodeat(temp_index).value.get_name()
            
            for i_n in range(temp_index):
                i = temp_index - 1 - i_n

                if self.node_list.nodeat(i).value.get_name() == name_to_look:
                    is_there_same_below = True
                    lesser_nodes.appendleft(self.node_list.nodeat(i).value)
                    break
                    #enough just once, can break here, save only the previous value
                if self.node_list.nodeat(i).value.get_name() > name_to_look:
                    break
                

            if not is_there_same_below:
                if not name_to_look.get_has_bigger():
                    last_change = self.node_list.nodeat(temp_index).value.values.last.value
                    break

            
                #name_to_look = self.node_list.nodeat(temp_index).value.get_name()
                lesser = name_to_look.get_lesser_from_bigger()
                bigger = name_to_look.get_bigger()

                is_lesser_from_bigger_there = False
                lesser_index = temp_index
                lesser_nodes.clear()

                for i_n in range(temp_index):
                    i = temp_index - 1 - i_n

                    if self.node_list.nodeat(i).value.get_name() == bigger:
                        break
                    if self.node_list.nodeat(i).value.get_name() > bigger:
                        break

                    if self.node_list.nodeat(i).value.get_name() == name_to_look:
                        is_lesser_from_bigger_there = False
                        break
                    if self.node_list.nodeat(i).value.get_name() > name_to_look:
                        is_lesser_from_bigger_there = False
                        break

                    if self.node_list.nodeat(i).value.get_name() == lesser:
                        is_lesser_from_bigger_there = True
                        lesser_index = i

                if not is_lesser_from_bigger_there:
                    last_change = self.node_list.nodeat(temp_index).value.values.last.value
                    break

                lesser_nodes.appendleft(self.node_list.nodeat(lesser_index).value)

                name_to_look = bigger
            else:
                name_to_look = name_to_look.get_incremented()

            is_there_already = False

            index_to_put = temp_index + 1
            while index_to_put < len(self.node_list):
                
                if self.node_list.nodeat(index_to_put).value.get_name() == name_to_look:
                    is_there_already = True
                    break
                if self.node_list.nodeat(index_to_put).value.get_name() > name_to_look:
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

            values = self.node_list.nodeat(i).value.get_values()
            for j, value in enumerate(values):
                value.derive_implication(lesser_values, j)
    
                
    def copy(self, copied = None):
        
        node_list = self.node_list

        if copied is None:
            copied = PatternReader()
            
            values = node_list.nodeat(0).value.get_values()
            for i in range(len(values)):
                copied.interpretation(values.nodeat(i).value.copy())


        for i in range(len(copied.node_list)):

            i_adapted = i
            while i_adapted < len(node_list) and not node_list.nodeat(i_adapted).value.get_name() == copied.node_list.nodeat(i).value.get_name():
                i_adapted += 1

            for j in range(len(copied.node_list.nodeat(i).value.values)):
                value = node_list.nodeat(i_adapted).value.values.nodeat(j).value.copy()
                copied.node_list.nodeat(i).value.values.nodeat(j).value = value
        
        return copied

    def __str__(self):
        return "\n".join(str(node) for node in self.node_list)



