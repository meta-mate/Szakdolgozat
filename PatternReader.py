from llist import dllist, dllistnode

class NodeValue:
    def derive_implication(self, node_values):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this!")


class PatternReader:
    class Node:
        def __init__(self, name, value=None):
            self.name = name
            self.value = value
            self.first_value = value
            self.values = dllist()
            self.has_bigger = False
            self.lesser_from_bigger = dllist()
            self.bigger = self.calculate_bigger()
            self.incremented = self.calculate_incremented()
    

            # Initialize values
            if value is not None:
                self.values.append(value)
                self.first_value = value


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

        def set_value(self, value):
            self.value = value
            if self.first_value is None:
                self.first_value = value
            self.values.append(value)

        def get_value(self):
            return self.value

        def get_values(self):
            return self.values

        def get_first_value(self):
            return self.first_value

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
                values_str += str(values[i])
                if i < len(values) - 1:
                    values_str += " | "
            return f"Node(Name: {self.get_name()}, Values: {values_str})"

    def __init__(self):
        self.node_list = dllist()
        self.pattern_length = 0

    def reset(self):
        self.node_list.clear()
        self.pattern_length = 0

    def interpretation(self, pattern):
        last_change = None
        self.pattern_length += 1

        def append_node(start_index, new_value):
            
            implication = None
            
            if start_index < len(self.node_list):
                implication = new_value.derive_implication(self.node_list.nodeat(start_index).value.get_values())
                self.node_list.nodeat(start_index).value.set_value(new_value)
                new_value = implication
                
                start_index += 1

            temp_index = start_index
            name_to_look = dllist([1])

            for i in range(start_index, len(self.node_list)):
                if self.node_list.nodeat(i).value.get_name() == name_to_look:
                    temp_index = i + 1
                    implication = new_value.derive_implication(self.node_list.nodeat(i).value.get_values())
                    self.node_list.nodeat(i).value.set_value(new_value)
                    new_value = implication
                else:
                    break

            if temp_index < len(self.node_list):
                self.node_list.insert(self.Node(dllist([1]), new_value), self.node_list.nodeat(temp_index))
            elif len(self.node_list):
                self.node_list.append(self.Node(dllist([1]), new_value))
            else:
                self.node_list.append(self.Node(dllist([0]), new_value))

            return temp_index

        temp_index = append_node(0, pattern)

        while True:
            
            is_there_same_below = False
            last_value_implication = None
            first_values = dllist()
            name_to_look = self.node_list.nodeat(temp_index).value.get_name()
            
            for i_n in range(temp_index):
                i = temp_index - 1 - i_n

                if self.node_list.nodeat(i).value.bigger_than(name_to_look):
                    break
                elif self.node_list.nodeat(i).value.name_equals(name_to_look):
                    is_there_same_below = True
                    first_values.appendleft(self.node_list.nodeat(i).value.get_first_value())

            if is_there_same_below == True:
                last_value_implication = self.node_list.nodeat(temp_index).value.get_first_value().derive_implication(first_values)

            if is_there_same_below == False:
                if self.node_list.nodeat(temp_index).value.get_has_bigger() == False:
                    last_change = self.node_list.nodeat(temp_index).value.get_value()
                    break

                name_to_look = self.node_list.nodeat(temp_index).value.get_lesser_from_bigger()

                is_lesser_from_bigger_there = False
                lesser_index = temp_index
                first_values.clear()

                while True:
                    for i_n in range(lesser_index):
                        i = lesser_index - 1 - i_n
                        if self.node_list.nodeat(i).value.bigger_than(name_to_look):
                            break
                        if self.node_list.nodeat(i).value.name_equals(name_to_look):
                            is_lesser_from_bigger_there = True
                            lesser_index = i

                    if is_lesser_from_bigger_there == False:
                        break
                    if lesser_index == temp_index:
                        break

                    first_values.appendleft(self.node_list.nodeat(lesser_index).value.get_first_value())
                    name_to_look = self.node_list.nodeat(lesser_index).value.get_lesser_from_bigger()

                    if self.node_list.nodeat(lesser_index).value.get_has_bigger() == False:
                        break

                    if len(name_to_look) <= 1:
                        if name_to_look.nodeat(0) <= 1:
                            break

                if is_lesser_from_bigger_there == True:
                    last_value_implication = self.node_list.nodeat(temp_index).value.get_first_value().derive_implication(first_values)
                
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

            if is_there_already == True:
                temp_index = append_node(index_to_put, last_value_implication)
            else:
                
                if index_to_put < len(self.node_list):
                    self.node_list.insert(self.Node(name_to_look, last_value_implication), self.node_list.nodeat(index_to_put))
                else:
                    self.node_list.append(self.Node(name_to_look, last_value_implication))

                temp_index = index_to_put
        
        return last_change


        if self.pattern_length <= 1:
            last_change = None

        return last_change

    def __str__(self):
        return "\n".join(str(node) for node in self.node_list)

# Example usage:
# Define a subclass of NodeValue with actual implementation
class MyNodeValue(NodeValue):
    def derive_implication(self, node_values):
        # Example implementation of derive_implication
        return self

    def __str__(self):
        return "NodeValue"


