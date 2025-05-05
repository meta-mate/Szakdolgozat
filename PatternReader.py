from llist import dllist, dllistnode

class NodeValue:
    def derive_implication(self, values, n):
        raise NotImplementedError("Subclasses should implement this!")

    def create_empty(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this!")


class PatternReader:
    class Node:
        def __init__(self, name, lesser_values=None):
            self.name = name
            self.values = dllist()
            
            if lesser_values is not None:
                self.lesser_values = lesser_values
            else:
                self.lesser_values = dllist()

            self.has_bigger = False
            self.lesser_from_bigger = dllist()
            self.bigger = self.calculate_bigger()
            self.incremented = self.calculate_incremented()


        def name_equals(self, other_name):
            name = self.name

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

        def calculate_incremented(self):
            r = dllist([node for node in self.name])
            r.insert(1, r.first.next)
            return r

        def calculate_bigger(self):
            r = dllist([1])
            self.lesser_from_bigger = dllist([1])

            name = self.name

            self.has_bigger = False

            if len(name) < 2:
                return r

            last_same_index = 0
            for i in range(2, len(name)):
                if name.nodeat(i).value == name.nodeat(i - 1).value:
                    last_same_index = i
                    self.lesser_from_bigger.append(name.nodeat(i).value)
                    self.has_bigger = True
                else:
                    break

            for i in range(last_same_index + 1, len(name)):
                r.append(name.nodeat(i).value)
                self.lesser_from_bigger.append(name.nodeat(i).value)

            if last_same_index != 0:
                r.insert(name.nodeat(last_same_index).value + 1, r.first.next)
                #(str(name) + " has bigger: " + str(r) + " so it has lesser: " + str(self.lesser_from_bigger))

            self.bigger = r
            return r

        def bigger_than(self, other_name):
            name = self.name
            
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

        #not needed from here

        def set_value(self, value):
            self.value = value
            if self.first_value is None:
                self.first_value = value
            self.values.append(value)

        def get_value(self):
            if len(self.values) <= 0:
                return None
            return self.values.nodeat(len(self.values) - 1).value

        def get_first_value(self):
            if len(self.values) <= 0:
                return None
            return self.values.nodeat(0).value

        #to here

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
            return self.incremented
        
        def copy_name(self):
            return dllist([node for node in self.name])

        def __str__(self):
            values_str = ""
            values = self.get_values()
            for i in range(len(values)):
                values_str += str(values.nodeat(i).value)
                if i < len(values) - 1:
                    values_str += " | "
            values_str = values_str.strip()
            values_str = values_str.replace("\n", "\n\t\t")

            lesser_values_str = ""
            lesser_values = self.get_lesser_values()
            for i in range(len(lesser_values)):
                lesser_values_str += str(lesser_values.nodeat(i).value)
                if i < len(lesser_values) - 1:
                    lesser_values_str += " | "
            lesser_values_str = lesser_values_str.strip()
            lesser_values_str = lesser_values_str.replace("\n", "\n\t\t")

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
                new_empty = self.node_list.nodeat(0).value.get_value().create_empty()
                values = self.node_list.nodeat(start_index).value.get_values()
                values.append(new_empty)

                start_index += 1

            temp_index = start_index
            name_to_look = dllist([1])

            for i in range(start_index, len(self.node_list)):
                if self.node_list.nodeat(i).value.get_name() == name_to_look:
                    temp_index = i + 1
                    #implication = new_value.derive_implication(self.node_list.nodeat(i).value.get_values())
                    #self.node_list.nodeat(i).value.set_value(new_value)
                    #new_value = implication

                    values = self.node_list.nodeat(i).value.get_values()
                    values.append(pattern.create_empty())
                else:
                    break

            if temp_index < len(self.node_list):
                self.node_list.insert(self.Node(dllist([1]), values), self.node_list.nodeat(temp_index))
                self.node_list.nodeat(temp_index).value.get_values().append(pattern.create_empty())
            elif len(self.node_list):
                self.node_list.append(self.Node(dllist([1]), values))
                self.node_list.nodeat(temp_index).value.get_values().append(pattern.create_empty())
            else:
                self.node_list.append(self.Node(dllist([0]), values))
                self.node_list.nodeat(temp_index).value.get_values().append(pattern.create_empty())

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

                if self.node_list.nodeat(i).value.bigger_than(name_to_look):
                    #also save if there is bigger below, so we know if this is the first
                    is_there_bigger_below = True
                    break
                elif self.node_list.nodeat(i).value.name_equals(name_to_look):
                    is_there_same_below = True
                    first_values.appendleft(self.node_list.nodeat(i).value.get_values().first.value)
                    break
                    #enough just once, can break here, save only the previous value

            #if is_there_same_below == True:
                #last_value_implication = self.node_list.nodeat(temp_index).value.get_first_value().derive_implication(first_values)
                #no need to calculate implication

            if is_there_same_below == False:
                if self.node_list.nodeat(temp_index).value.get_has_bigger() == False:
                    last_change = self.node_list.nodeat(temp_index).value.get_value()
                    break

            
                #name_to_look = self.node_list.nodeat(temp_index).value.get_name()
                lesser_name = self.node_list.nodeat(temp_index).value.get_lesser_from_bigger()

                is_lesser_from_bigger_there = False
                lesser_index = temp_index
                first_values.clear()

                for i_n in range(temp_index):
                    i = temp_index - 1 - i_n
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
                
                if self.node_list.nodeat(index_to_put).value.bigger_than(name_to_look):
                    break
                elif self.node_list.nodeat(index_to_put).value.name_equals(name_to_look):
                    is_there_already = True
                    break

                index_to_put += 1

            first_values.append(self.node_list.nodeat(temp_index).value.get_values().first.value)

            if is_there_already == True:
                temp_index = append_node(index_to_put, first_values.last.value)
            else:
                if index_to_put < len(self.node_list):
                    self.node_list.insert(self.Node(name_to_look, first_values), self.node_list.nodeat(index_to_put))
                    self.node_list.nodeat(index_to_put).value.get_values().append(pattern.create_empty())
                else:
                    self.node_list.append(self.Node(name_to_look, first_values))
                    self.node_list.nodeat(index_to_put).value.get_values().append(pattern.create_empty())

                temp_index = index_to_put
        
        return last_change


    def calculate_values(self):
        for j in range(1, len(self.node_list)):
            for i, value in enumerate(self.node_list.nodeat(j).value.get_values()):
                lesser_values = self.node_list.nodeat(j).value.get_lesser_values()
                value.derive_implication(lesser_values, i)
                

    def __str__(self):
        return "\n".join(str(node) for node in self.node_list)



